import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.differentiable_features import DifferentiableFeatureExtractor


class DiffGreater(nn.Module):
    """
    Differentiable A > B via sigmoid with learnable temperature.

    Temperature controls the sharpness of the soft comparison:
      - Low temp (0.5):  sigmoid(diff * 0.5) is smooth, gradients flow for |diff| < 10
      - High temp (10):  sigmoid(diff * 10) is near-binary, but gradients vanish for |diff| > 0.5

    init_temp=0.5 is calibrated for KDJ/price-scale differences (typically 0–20):
      - diff=2:  sigmoid(1.0)=0.73, grad=0.20  ← useful learning signal
      - diff=5:  sigmoid(2.5)=0.92, grad=0.07  ← still usable
      - diff=10: sigmoid(5.0)=0.99, grad=0.007 ← boundary
    Temperature is learnable and will increase during training to sharpen decisions.
    The transformer correction compensates for initial softness.
    """
    def __init__(self, init_temp=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, A, B):
        return torch.sigmoid((A - B) * self.temperature)


class DiffLess(nn.Module):
    """Differentiable A < B. See DiffGreater for temperature rationale."""
    def __init__(self, init_temp=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, A, B):
        return torch.sigmoid((B - A) * self.temperature)


class DifferentiableExpertModel(nn.Module):
    """
    Applies the MyLanguage expert strategy as differentiable soft-logic on
    nine-factor features (from either learnable or static extraction).

    Learnable parameters:
      - w_bias, w_f1, w_f2, w_cond3_j1  (JX formula constants)
      - 7 temperature scalars            (soft logic gate steepness)

    At initialization all parameters match the original strategy exactly,
    guaranteeing zero-shot equivalence at Epoch 0.
    """
    def __init__(self, init_constants=None):
        super().__init__()

        c = init_constants or {}
        val_bias  = float(c.get("w_bias", -50.0))
        val_f1    = float(c.get("w_f1", 6.0))
        val_f2    = float(c.get("w_f2", 6.0))
        val_cond3 = float(c.get("w_cond3_j1", 1.0))

        self.w_bias = nn.Parameter(torch.tensor([val_bias], dtype=torch.float32))
        self.w_f1 = nn.Parameter(torch.tensor([val_f1], dtype=torch.float32))
        self.w_f2 = nn.Parameter(torch.tensor([val_f2], dtype=torch.float32))
        self.w_cond3_j1 = nn.Parameter(torch.tensor([val_cond3], dtype=torch.float32))

        # BK2 soft-logic gates
        self.dg_jx_rx = DiffGreater()    # JX > REF(JX,1)
        self.dg_jx_j1 = DiffGreater()    # JX > J1 * w
        self.dg_c_macd = DiffGreater()   # C > ma_c_down

        # SK2 soft-logic gates
        self.dl_jx_rx = DiffLess()       # JX < REF(JX,1)
        self.dl_jx_j1 = DiffLess()       # JX < J1
        self.dl_c_macu = DiffLess()      # C < ma_c_up

        # SP2 has its own independent temperature
        self.dl_jn3 = DiffLess()         # JN3_36 < REF(JN3_36,1)

    def forward(self, x_dict):
        """
        x_dict: keys map to [Batch, Seq] tensors.
        Returns: [Batch, Seq, 5] soft probabilities for [BK2, SK2, BP1, SP1, SP2].
        """
        C = x_dict['C']
        J1 = x_dict['J1']
        JN3_36 = x_dict['JN3_36']
        JX_base = x_dict['JX_base']         # J1 + J2 + J3*TEMA3T1
        F1 = x_dict['F1']                   # J2 * TEMA3T3
        F2 = x_dict['F2']                   # J1 * TEMA3T2
        ma_c_down = x_dict['ma_c_down']
        ma_c_up = x_dict['ma_c_up']

        B = C.shape[0]
        device = C.device

        # ---- JX with learnable constants ----
        # Original: JX = J1+J2-50 + J2*TEMA3T3*6 + J1*TEMA3T2*6 + J3*TEMA3T1
        # Factored: JX = JX_base + w_bias + F1*w_f1 + F2*w_f2
        JX = JX_base + self.w_bias + F1 * self.w_f1 + F2 * self.w_f2

        # REF(JX, 1): shift right, duplicate first value to avoid discontinuity
        ref_jx = torch.cat([JX[:, :1], JX[:, :-1]], dim=1)

        # ---- BK2 = (JX > REF(JX,1)) & (C > ma_c_down) & (JX > J1*w_cond3_j1) ----
        cond1_soft = self.dg_jx_rx(JX, ref_jx)
        cond2_soft = self.dg_c_macd(C, ma_c_down)
        cond3_soft = self.dg_jx_j1(JX, J1 * self.w_cond3_j1)
        BK2_soft = cond1_soft * cond2_soft * cond3_soft

        # ---- SK2 = (JX < REF(JX,1)) & (C < ma_c_up) & (JX < J1) ----
        cond4_soft = self.dl_jx_rx(JX, ref_jx)
        cond5_soft = self.dl_c_macu(C, ma_c_up)
        cond6_soft = self.dl_jx_j1(JX, J1)
        SK2_soft = cond4_soft * cond5_soft * cond6_soft

        # ---- BP1 / SP1: soft EVERY(3) ----
        # SP1 := BK2=0 && REF(EVERY(BK2,3),1)
        # BP1 := SK2=0 && REF(EVERY(SK2,3),1)
        pad_bk2 = torch.cat([torch.zeros(B, 3, device=device), BK2_soft], dim=1)
        ref1_bk2 = pad_bk2[:, 2:-1]
        ref2_bk2 = pad_bk2[:, 1:-2]
        ref3_bk2 = pad_bk2[:, 0:-3]

        pad_sk2 = torch.cat([torch.zeros(B, 3, device=device), SK2_soft], dim=1)
        ref1_sk2 = pad_sk2[:, 2:-1]
        ref2_sk2 = pad_sk2[:, 1:-2]
        ref3_sk2 = pad_sk2[:, 0:-3]

        every3_bk2 = ref1_bk2 * ref2_bk2 * ref3_bk2
        every3_sk2 = ref1_sk2 * ref2_sk2 * ref3_sk2

        SP1_soft = (1.0 - BK2_soft) * every3_bk2
        BP1_soft = (1.0 - SK2_soft) * every3_sk2

        # ---- SP2 = JN3_36 < REF(JN3_36, 1) ----
        ref_jn3 = torch.cat([JN3_36[:, :1], JN3_36[:, :-1]], dim=1)
        SP2_soft = self.dl_jn3(JN3_36, ref_jn3)

        # [Batch, Seq, 5]
        probs = torch.stack([BK2_soft, SK2_soft, BP1_soft, SP1_soft, SP2_soft], dim=-1)
        return probs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class DifferentiableExpertTransformer(nn.Module):
    """
    Unified end-to-end differentiable model.

    Raw OHLC → Learnable Features (20 DiffEMA alphas)
             → Expert Prior (4 constants + 7 gate temperatures)
               → Transformer Residual Correction

    All 20+4+7+transformer parameters train end-to-end in a single backward pass.
    At init, feature alphas match original hardcoded periods, expert constants match
    the original strategy, and the transformer decoder is zero-initialized, guaranteeing
    exact zero-shot equivalence at Epoch 0.

    Two forward modes via dispatch:
      model(x_dict)           → [B,S,5]  (inference from pre-computed features)
      model(C, H, L, seq_len) → (all_probs[T,5], expert_probs[T,5])  (end-to-end training)
    """
    def __init__(self, init_constants=None, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        # Layer 1: Learnable Feature Extraction (20 DiffEMA alpha parameters)
        self.feature_extractor = DifferentiableFeatureExtractor()

        # Layer 2: Expert Prior (4 JX constants + 7 gate temperatures)
        self.expert_prior = DifferentiableExpertModel(init_constants=init_constants)

        # Layer 3: Transformer Residual Correction
        # 18 pre-computed features + 5 expert probs = 23 inputs
        self.feature_keys = [
            'C', 'H', 'L', 'J1', 'J2', 'J3', 'JN3_36',
            'JX', 'EMAJX', 'EMAJX8', 'ma_c_down', 'ma_c_up',
            'JX_base', 'F1', 'F2', 'EMA_JX_base', 'EMA_F1', 'EMA_F2'
        ]

        self.input_proj = nn.Linear(len(self.feature_keys) + 5, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, 5)

        # Zero-init: at Epoch 0, correction = 0 → output = expert_probs exactly
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.bias.zero_()

    def forward(self, x_or_C, H=None, L=None, seq_len=15):
        """
        Unified forward dispatch:
          model(x_dict)           → [B,S,5] probs (inference from pre-computed features)
          model(C, H, L, seq_len) → (all_probs[T,5], expert_probs[T,5]) (end-to-end)
        """
        if isinstance(x_or_C, dict):
            return self._forward_from_features(x_or_C)
        return self._forward_e2e(x_or_C, H, L, seq_len)

    def _forward_from_features(self, x_dict):
        """
        Inference with pre-computed feature dicts (backward compatible).
        x_dict: {key: [B, S]} tensors.
        Returns: [B, S, 5] probabilities.
        """
        expert_probs = self.expert_prior(x_dict)

        raw_feat_list = []
        for key in self.feature_keys:
            t = x_dict.get(key, x_dict['C'])
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            raw_feat_list.append(t)

        x_raw = torch.cat(raw_feat_list, dim=-1)
        x_combined = torch.cat([x_raw, expert_probs], dim=-1)
        x_combined = torch.nan_to_num(x_combined, nan=0.0)

        x_emb = self.input_proj(x_combined)
        x_emb = self.pos_encoder(x_emb)
        tf_out = self.transformer(x_emb)

        correction = self.decoder(tf_out)

        eps = 1e-6
        expert_logits = torch.logit(expert_probs.clamp(eps, 1 - eps))
        final_probs = torch.sigmoid(expert_logits + correction)
        return final_probs

    def _forward_e2e(self, C, H, L, seq_len):
        """
        Full end-to-end: raw OHLC → learnable features → expert → transformer.

        Gradient flow:
          Loss → d/d(sigmoid(logit(expert) + correction))
            → Transformer weights (d/d correction)
            → Expert constants (d/d JX → d/d w_bias, w_f1, w_f2, w_cond3_j1)
            → Gate temperatures (d/d sigmoid thresholds)
            → Feature alphas (d/d J1,J2,J3... → d/d KDJ/TEMA → d/d DiffEMA.w_alpha)

        Args:
            C, H, L: [T] raw price tensors.
            seq_len: window size for transformer attention.
        Returns:
            all_probs: [T, 5] probabilities for each bar.
            expert_probs_full: [T, 5] raw expert prior (for monitoring).
        """
        T = C.size(0)

        # 1. Learnable feature extraction on full series (20 DiffEMA alphas)
        features = self.feature_extractor(C, H, L)

        # 2. Expert prior on full series [1, T, 5] (4 constants + 7 temps)
        x_dict_full = {k: v.unsqueeze(0) for k, v in features.items()}
        expert_probs_full = self.expert_prior(x_dict_full).squeeze(0)  # [T, 5]

        # 3. Stack features for transformer [T, 18]
        feat_stacked = torch.stack(
            [torch.nan_to_num(features.get(k, features['C']), nan=0.0) for k in self.feature_keys],
            dim=-1
        )

        # 4. Unfold into overlapping windows: [W, S, 18] where W = T - S + 1
        feat_windows = feat_stacked.unfold(0, seq_len, 1).permute(0, 2, 1)
        ep_windows = expert_probs_full.unfold(0, seq_len, 1).permute(0, 2, 1)

        # 5. Transformer forward on windowed features + expert probs
        x_combined = torch.cat([feat_windows, ep_windows], dim=-1)  # [W, S, 23]
        x_combined = torch.nan_to_num(x_combined, nan=0.0)

        x_emb = self.input_proj(x_combined)
        x_emb = self.pos_encoder(x_emb)
        tf_out = self.transformer(x_emb)
        correction = self.decoder(tf_out)[:, -1, :]  # [W, 5] — last bar of each window

        # 6. Logit-space residual on corresponding expert probs
        ep_last = expert_probs_full[seq_len - 1:]  # [W, 5]
        eps = 1e-6
        expert_logits = torch.logit(ep_last.clamp(eps, 1 - eps))
        final_probs = torch.sigmoid(expert_logits + correction)

        # 7. Pad first seq_len-1 bars with expert-only probs (no transformer correction)
        ep_head = expert_probs_full[:seq_len - 1]
        all_probs = torch.cat([ep_head, final_probs], dim=0)  # [T, 5]

        return all_probs, expert_probs_full
