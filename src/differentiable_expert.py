import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffGreater(nn.Module):
    """
    Differentiable A > B via a steep sigmoid function.
    At initialization (temperature=100.0), it strictly mimics hard logic A > B.
    During training, the threshold or temperature can be learned.
    """
    def __init__(self, init_temp=100.0):
        super().__init__()
        # Make the steepness parameterizable so the network can soften the logic if needed
        self.temperature = nn.Parameter(torch.tensor([init_temp]))
        
    def forward(self, A, B):
        # returns approx 1.0 if A > B, 0.0 if A < B
        return torch.sigmoid((A - B) * self.temperature)

class DiffLess(nn.Module):
    """ Differentiable A < B """
    def __init__(self, init_temp=100.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))
        
    def forward(self, A, B):
        # returns approx 1.0 if A < B, 0.0 if A > B
        return torch.sigmoid((B - A) * self.temperature)

class DiffCrossDown(nn.Module):
    """
    Differentiable equivalent of CROSSDOWN(A, B).
    True if A_{t-1} >= B_{t-1} AND A_t < B_t.
    """
    def __init__(self, init_temp=100.0):
        super().__init__()
        self.temp_prev = nn.Parameter(torch.tensor([init_temp]))
        self.temp_curr = nn.Parameter(torch.tensor([init_temp]))
        
    def forward(self, A, B):
        # Soft A_{t-1} >= B_{t-1}
        A_prev = torch.cat([A[:, :1]*0, A[:, :-1]], dim=1)
        B_prev = torch.cat([B[:, :1]*0, B[:, :-1]], dim=1)
        prev_gte = torch.sigmoid((A_prev - B_prev) * self.temp_prev)
        
        # Soft A_t < B_t
        curr_lt = torch.sigmoid((B - A) * self.temp_curr)
        
        # Soft AND operation (multiplication)
        return prev_gte * curr_lt

class DiffCrossUp(nn.Module):
    """
    Differentiable equivalent of CROSS(A, B).
    True if A_{t-1} <= B_{t-1} AND A_t > B_t
    """
    def __init__(self, init_temp=100.0):
        super().__init__()
        self.temp_prev = nn.Parameter(torch.tensor([init_temp]))
        self.temp_curr = nn.Parameter(torch.tensor([init_temp]))
        
    def forward(self, A, B):
        A_prev = torch.cat([A[:, :1]*0, A[:, :-1]], dim=1)
        B_prev = torch.cat([B[:, :1]*0, B[:, :-1]], dim=1)
        prev_lte = torch.sigmoid((B_prev - A_prev) * self.temp_prev)
        
        curr_gt = torch.sigmoid((A - B) * self.temp_curr)
        return prev_lte * curr_gt

class DifferentiableExpertModel(nn.Module):
    def __init__(self, init_constants=None):
        super().__init__()
        
        # We start by initializing the "Expert" combination layer to precisely match
        # the parameters of the human expert strategy.
        # Original: JX:=J1+J2-50+J2*TEMA3T3*6+J1*TEMA3T2*6+J3*TEMA3T1;
        
        # Load custom AST constants or fallback to original Human Expert baselines
        c = init_constants or {}
        val_bias  = float(c.get("w_bias", -50.0))
        val_f1    = float(c.get("w_f1", 6.0))
        val_f2    = float(c.get("w_f2", 6.0))
        val_cond3 = float(c.get("w_cond3_j1", 1.0))
        
        # Register as PyTorch parameters with requires_grad=True
        # We explicitly cast to float() because JSON payloads might truncate 6.0 into int 6,
        # which creates a torch.int64 tensor incapable of `requires_grad=True`
        self.w_bias = nn.Parameter(torch.tensor([val_bias], dtype=torch.float32))
        self.w_f1 = nn.Parameter(torch.tensor([val_f1], dtype=torch.float32))
        self.w_f2 = nn.Parameter(torch.tensor([val_f2], dtype=torch.float32))
        
        # The scaling factor for Cond3: JX > J1 * w
        self.w_cond3_j1 = nn.Parameter(torch.tensor([val_cond3], dtype=torch.float32))
        
        # Differentiable Logic gates
        self.dg_jx_rx = DiffGreater()
        self.dg_jx_j1 = DiffGreater()
        self.dg_c_macd = DiffGreater()
        
        self.dl_jx_rx = DiffLess()
        self.dl_jx_j1 = DiffLess()
        self.dl_c_macu = DiffLess()
        
    def forward(self, x_dict):
        """
        Takes in a dictionary of continuous tensors (like C, J1).
        Shape of each: [Batch, Sequence_Len]
        """
        C = x_dict['C']
        J1 = x_dict['J1']
        
        # We no longer accept JX and EMAJX passively.
        # We construct them DYNAMICALLY using our learnable constants.
        JX_base = x_dict['JX_base']
        F1 = x_dict['F1']
        F2 = x_dict['F2']
        
        EMA_JX_base = x_dict['EMA_JX_base']
        EMA_F1 = x_dict['EMA_F1']
        EMA_F2 = x_dict['EMA_F2']
        
        # Dynamic Native Evaluation using GPUs parameterized scales
        JX = JX_base + self.w_bias + F1 * self.w_f1 + F2 * self.w_f2
        
        # Due to EMA linearity: 
        # EMA(A + B + C*D) = EMA(A) + EMA(B) + D*EMA(C) (if D is a constant over time)
        # Therefore, because our parameter W_bias, W_f1, W_f2 are constant across the sequence window:
        EMAJX = EMA_JX_base + self.w_bias + EMA_F1 * self.w_f1 + EMA_F2 * self.w_f2
        
        # Get the precomputed dynamic constraints
        ma_c_down = x_dict['ma_c_down']
        ma_c_up = x_dict['ma_c_up']
        
        # ------------ BK2 Logic ------------
        # cond1 = JX > REF(JX, 1)
        ref_jx = torch.cat([JX[:, :1], JX[:, :-1]], dim=1) # REF 1
        cond1_soft = self.dg_jx_rx(JX, ref_jx)
        
        # cond3 = JX > J1 * w
        cond3_soft = self.dg_jx_j1(JX, J1 * self.w_cond3_j1)
        
        # cond2 = C > ma_c_down
        cond2_soft = self.dg_c_macd(C, ma_c_down)
        
        # BK2 = cond1 & cond2 & cond3
        BK2_soft = cond1_soft * cond2_soft * cond3_soft
        
        # ------------ SK2 Logic ------------
        cond4_soft = self.dl_jx_rx(JX, ref_jx)
        cond6_soft = self.dl_jx_j1(JX, J1)
        cond5_soft = self.dl_c_macu(C, ma_c_up)
        
        SK2_soft = cond4_soft * cond5_soft * cond6_soft
        
        # ------------ BP1 / SP1 Logic ------------
        # SP1:=BK2=0&&REF(EVERY(BK2,3),1);
        # SP1 is basically: did BK2 fire in the last 3 days consecutively but NOT today.
        # Since BK2_soft is continuous [0,1], EVERY(3) is approximately BK2_t-1 * BK2_t-2 * BK2_t-3
        # And BK2=0 is approximately (1.0 - BK2_t0)
        
        # Shift BK2 and SK2 back in time
        pad_bk2 = torch.cat([torch.zeros(C.shape[0], 3, device=C.device), BK2_soft], dim=1)
        ref1_bk2 = pad_bk2[:, 2:-1]
        ref2_bk2 = pad_bk2[:, 1:-2]
        ref3_bk2 = pad_bk2[:, 0:-3]
        
        pad_sk2 = torch.cat([torch.zeros(C.shape[0], 3, device=C.device), SK2_soft], dim=1)
        ref1_sk2 = pad_sk2[:, 2:-1]
        ref2_sk2 = pad_sk2[:, 1:-2]
        ref3_sk2 = pad_sk2[:, 0:-3]
        
        # Soft EVERY(3)
        every3_bk2_ref1 = ref1_bk2 * ref2_bk2 * ref3_bk2
        every3_sk2_ref1 = ref1_sk2 * ref2_sk2 * ref3_sk2
        
        # Soft SP1 and BP1
        SP1_soft = (1.0 - BK2_soft) * every3_bk2_ref1
        BP1_soft = (1.0 - SK2_soft) * every3_sk2_ref1
        
        # ------------ SP2 Logic ------------
        # SP2:= JN3_36 < REF(JN3_36, 1)
        JN3_36 = x_dict['JN3_36']
        ref_jn3 = torch.cat([JN3_36[:, :1], JN3_36[:, :-1]], dim=1)
        # We reuse the soft less-than logic originally designed for JX
        SP2_soft = self.dl_jx_rx(JN3_36, ref_jn3) 
        
        # Output [Batch, Seq, 5]
        return torch.stack([BK2_soft, SK2_soft, BP1_soft, SP1_soft, SP2_soft], dim=-1)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class DifferentiableExpertTransformer(nn.Module):
    def __init__(self, init_constants=None, num_features=18, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.expert_prior = DifferentiableExpertModel(init_constants=init_constants)
        
        self.feature_keys = ['C', 'H', 'L', 'J1', 'J2', 'J3', 'JN3_36', 'JX', 'EMAJX', 'EMAJX8', 'ma_c_down', 'ma_c_up', 'JX_base', 'F1', 'F2', 'EMA_JX_base', 'EMA_F1', 'EMA_F2']
        
        # Feature processing for transformer
        # We ingest 18 keys from x_dict + 5 signal probabilities
        self.input_proj = nn.Linear(len(self.feature_keys) + 5, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, 5)
        
        # ZERO INITIALIZATION
        # Ensuring Epoch 0 is utterly identical to human expert equations
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.bias.zero_()
        
    def forward(self, x_dict):
        # 1. Human Logic Prior Probabilities
        expert_probs = self.expert_prior(x_dict) # [Batch, Seq, 5]
        
        # 2. Extract RAW Input sequence from dict values
        # By iterating reliably through the enforced 18-tensor keys list
        raw_feat_list = []
        for key in self.feature_keys:
            t = x_dict.get(key, x_dict['C'])
            if t.dim() == 2:
                t = t.unsqueeze(-1) # [Batch, Seq, 1]
            raw_feat_list.append(t)
            
        x_raw = torch.cat(raw_feat_list, dim=-1) # [Batch, Seq, 18]
        
        # Combine [Features, PriorProbs] -> [Batch, Seq, 23]
        x_combined = torch.cat([x_raw, expert_probs.detach()], dim=-1)
        
        # CRISIS FIX: Prevent leading Moving-Average NaNs from poisoning the Global Self-Attention matrix
        x_combined = torch.nan_to_num(x_combined, nan=0.0)
        
        # 3. Transformer Forward
        x_emb = self.input_proj(x_combined)
        x_emb = self.pos_encoder(x_emb)
        tf_out = self.transformer(x_emb)
        
        # 4. Residual Correction
        # The zero-initialized decoder will only output 0.0 initially, leaving expert_probs intact
        correction = self.decoder(tf_out) 
        
        final_probs = torch.clamp(expert_probs + correction, 0.0, 1.0)
        
        # Optionally, one could apply sigmoid to correction and blend, but additive residual
        # on probability space clamped is a direct zero-shot guarantee.
        return final_probs
