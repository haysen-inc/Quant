import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import deque
from src.differentiable_expert import DifferentiableExpertTransformer
from src.train import AsymmetricExpertLoss
from src.features_torch import extract_features
from src.labels_torch import extract_labels

class OnlineRLAgent:
    def __init__(self, model_path='models/best_expert_1h.pth', hold_period=35, learning_rate=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load the best historical baseline model
        self.model = DifferentiableExpertTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.train()  # Online agent is ALWAYS in training mode
        
        # RL Hyperparameters
        self.hold_period = hold_period
        
        # We use a much smaller learning rate for online fine-tuning to avoid catastrophic forgetting 
        # of the core human expert structural layout
        expert_constants = ['expert_prior.w_bias', 'expert_prior.w_f1', 'expert_prior.w_f2', 'expert_prior.w_cond3_j1']
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if n in expert_constants], 'lr': learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if n not in expert_constants], 'lr': learning_rate * 0.1}
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        
        # Our primary asymmetric loss function linking the Expert BCELoss with actual Future Log Return scalar multipliers
        self.criterion = AsymmetricExpertLoss(lambda_profit=10.0, cost_fee=0.0)
        
        # The Pending Queue: Resolves the "Delayed Reward" problem
        # Stores tuples: (timestamp, execution_price, x_dict_state, y_expert_target)
        self.pending_queue = deque()
        self.metrics = {'trades_resolved': 0, 'cumulative_pnl': 0.0, 'win_count': 0}

    def process_step(self, current_timestamp, current_price, x_dict_tensor, y_target_tensor):
        """
        Processes a single streaming candle.
        1. Emits a live prediction based on the current state.
        2. Places the state into the pending queue.
        3. Dynamically resolves mature states from the queue when Exit signals trigger.
        """
        # 1. Forward Pass (Live Prediction)
        with torch.no_grad():
            self.model.eval()
            logits_seq = self.model(x_dict_tensor)
            probs = logits_seq[:, -1, :] # [1, 5] specific to current bar
            
        p = probs[0].cpu().numpy()
        y_val_cpu = y_target_tensor[0].cpu().numpy()
        
        # We allow either the RL Agent's soft exit OR the Expert's hard exit to resolve the queue
        # This prevents the queue from stalling infinitely if the un-trained agent is too shy to act
        long_exit = p[3] > 0.5 or p[4] > 0.5 or y_val_cpu[3] == 1.0 or y_val_cpu[4] == 1.0
        short_exit = p[2] > 0.5 or y_val_cpu[2] == 1.0
        
        # 2. Add to Pending Queue for delayed reward resolution
        self.pending_queue.append({
            'timestamp': current_timestamp,
            'entry_price': current_price,
            'x_dict': x_dict_tensor,
            'y_target': y_target_tensor,
            'r_long_resolved': False,
            'r_short_resolved': False,
            'r_long_val': 0.0,
            'r_short_val': 0.0
        })
        
        # 3. Resolve Mature States & Reinforce
        for state in self.pending_queue:
            if long_exit and not state['r_long_resolved']:
                state['r_long_val'] = np.log(current_price / state['entry_price'])
                state['r_long_resolved'] = True
            if short_exit and not state['r_short_resolved']:
                state['r_short_val'] = np.log(current_price / state['entry_price'])
                state['r_short_resolved'] = True
                
        # 3.5 Failsafe: Force resolution if a state has been held for an absurdly long time (>100 bars)
        if len(self.pending_queue) > 100:
            if not self.pending_queue[0]['r_long_resolved']:
                self.pending_queue[0]['r_long_val'] = np.log(current_price / self.pending_queue[0]['entry_price'])
                self.pending_queue[0]['r_long_resolved'] = True
            if not self.pending_queue[0]['r_short_resolved']:
                self.pending_queue[0]['r_short_val'] = np.log(current_price / self.pending_queue[0]['entry_price'])
                self.pending_queue[0]['r_short_resolved'] = True
                
        # If the oldest state in the queue has both returns fully resolved, pop and reinforce
        while self.pending_queue and self.pending_queue[0]['r_long_resolved'] and self.pending_queue[0]['r_short_resolved']:
            mature_state = self.pending_queue.popleft()
            self._reinforce(mature_state)
            
        return p

    def _reinforce(self, mature_state):
        """
        Updates the model gradient once an entry state is fully resolved by subsequent exits.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x_dict = mature_state['x_dict']
        y_target = mature_state['y_target']
        
        r_fwd_dual = torch.tensor([[mature_state['r_long_val'], mature_state['r_short_val']]], dtype=torch.float32, device=self.device)
        
        # Re-forward pass the state to obtain differentiable probabilities
        logits_seq = self.model(x_dict)
        probs = logits_seq[:, -1, :] # [1, 5]
        
        # Compute the live Asymmetric Expert Loss
        loss, alpha_realized, _ = self.criterion(probs, y_target, r_fwd_dual)
        
        # Reinforcement Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.metrics['trades_resolved'] += 1

def run_live_simulation(raw_code=None):
    """
    Test harness that simulates a live data stream using recent historical data.
    """
    import datetime
    from src.data_loader import fetch_spy_data
    from src.mylanguage_parser import parse_mylanguage
    
    print("Initiating Live Online RL Simulation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process Custom Rules via AST Engine dynamically
    ast_config = None
    if raw_code:
        ast_config = parse_mylanguage(raw_code)
        print(f"--- RL Agent Configured from User AST: {ast_config} ---")
    
    # Fetch completely unseen recent data (e.g., last 90 days)
    today = datetime.datetime.now()
    start_date = (today - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
    df = fetch_spy_data(start_date=start_date, end_date=None, interval="1h")
    print(f"Loaded {len(df)} live streaming candles.")
    
    # Critical Fix: Drop the warm-up period rows that naturally contain NaNs
    df.dropna(inplace=True)
    
    features_dict = extract_features(df)
    labels_dict = extract_labels(df, features_dict, ast_config=ast_config)
    
    # Deep Immunization: Some indicators may still spawn NaNs or Infs depending on edge cases.
    for k in features_dict:
        if isinstance(features_dict[k], torch.Tensor):
            features_dict[k] = torch.nan_to_num(features_dict[k], nan=0.0, posinf=0.0, neginf=0.0)
    for k in labels_dict:
        if isinstance(labels_dict[k], torch.Tensor):
            labels_dict[k] = torch.nan_to_num(labels_dict[k], nan=0.0, posinf=0.0, neginf=0.0)
    
    # Instantiate the Agent
    agent = OnlineRLAgent(model_path='models/best_expert_1h.pth', hold_period=35, learning_rate=0.01)
    
    # Real-World Strict No-Overlap Trackers
    rl_state = {'pos': 0, 'entry': 0, 'bars': 0, 'pnl': 0.0, 'wins': 0, 'trades': 0}
    base_state = {'pos': 0, 'entry': 0, 'bars': 0, 'pnl': 0.0, 'wins': 0, 'trades': 0}
    
    rl_pnl_history = []
    base_pnl_history = []
    rl_decision_history = [] # Tracks the combined final decision probabilities [Long Prob, Short Prob]
    
    trailing_probs = []
    
    seq_len = 15
    feature_keys=[
        'C', 'H', 'L', 
        'TEMA3T1', 'TEMA3T3', 'TEMA3T2', 'J1', 'J2', 'J3', 'JN3_36',
        'JX', 'EMAJX', 'EMAJX8', 'ma_c_down', 'ma_c_up',
        'JX_base', 'F1', 'F2', 'EMA_JX_base', 'EMA_F1', 'EMA_F2'
    ]
    label_keys=['BK2', 'SK2', 'BP1', 'SP1', 'SP2']
    
    print("Streaming data to Online RL Agent...")
    for i in range(seq_len, len(df)):
        # Construct Sequence State for current timestep cleanly from keys
        x_dict_tensor = {}
        for key in feature_keys:
            # Slicing the 1D feature array to get the (seq_len, ) window
            arr = np.array(list(features_dict[key][i-seq_len:i]), dtype=np.float32)
            # Add batch dimension [1, seq_len]
            x_dict_tensor[key] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Add required derived columns missing from the old RL script arrays
        for k in ['C', 'H', 'L', 'EMAJX', 'ma_c_down', 'ma_c_up', 'JX_base', 'F1', 'F2', 'EMA_JX_base', 'EMA_F1', 'EMA_F2']:
            if k not in x_dict_tensor and k in features_dict:
                arr = np.array(list(features_dict[k][i-seq_len:i]), dtype=np.float32)
                x_dict_tensor[k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Construct Expert Label for current timestep
        # labels_dict contains PyTorch Tensors, standard indexing applies
        y_arr = [labels_dict[k][i-1].item() for k in label_keys]
        y_target_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(0).to(device) # [1, 4]
        
        current_time = df.index[i-1]
        current_price = df['Close'].iloc[i-1]
        
        # Step the Agent! (Handles internal learning Queue)
        probs = agent.process_step(current_time, current_price, x_dict_tensor, y_target_tensor)
        
        # Adaptive 90th Percentile Thresholding (since Expert Loss violently crushes trailing probabilities)
        trailing_probs.append(probs)
        if len(trailing_probs) > 100: trailing_probs.pop(0)
        
        # Default safety threshold
        th_long = 0.5; th_short = 0.5
        if len(trailing_probs) >= 50:
            hist_p = np.array(trailing_probs)
            # Use strict rolling tops, clamped between [0.1, 0.8] to prevent total lockups from singular 1.0 spikes
            raw_top_long = np.percentile(np.maximum(hist_p[:,0], hist_p[:,2]), 95)
            raw_top_short = np.percentile(np.maximum.reduce([hist_p[:,1], hist_p[:,3], hist_p[:,4]]), 95)
            th_long = min(0.8, max(0.1, raw_top_long))
            th_short = min(0.8, max(0.1, raw_top_short))
            
        p_long = float(max(probs[0], probs[2]))
        p_short = float(max(probs[1], probs[3], probs[4]))
        rl_decision_history.append([p_long, p_short])
        
        # 2. Baseline Human Expert
        y_v = y_target_tensor[0].cpu().numpy()
        
        # --- Strict Non-Overlapping Portfolio Evaluator ---
        # 1. RL Agent
        if rl_state['pos'] == 0:
            if probs[0] >= th_long or probs[2] >= th_long:
                rl_state['pos'] = 1; rl_state['entry'] = current_price
            elif probs[1] >= th_short or probs[3] >= th_short or probs[4] >= th_short:
                rl_state['pos'] = -1; rl_state['entry'] = current_price
        else:
            rl_state['bars'] += 1
            # --- FAT TAIL ISOLATION FIX ---
            # The RL Agent should NOT be forced out of a perfectly good trend just because the noisy human baseline (y_v) 
            # randomly triggered an SP2 oscillator tick. It must rely completely on its own `probs` or its `hold_period`.
            is_long_exit_trigger = probs[3] >= th_short or probs[4] >= th_short or rl_state['bars'] >= agent.hold_period
            is_short_exit_trigger = probs[2] >= th_long or rl_state['bars'] >= agent.hold_period
            
            if (rl_state['pos'] == 1 and is_long_exit_trigger) or (rl_state['pos'] == -1 and is_short_exit_trigger):
                r = np.log(current_price / rl_state['entry'])
                rl_state['trades'] += 1
                if rl_state['pos'] == 1:
                    rl_state['pnl'] += r
                    if r > 0: rl_state['wins'] += 1
                elif rl_state['pos'] == -1:
                    rl_state['pnl'] -= r
                    if r < 0: rl_state['wins'] += 1
                rl_state['pos'] = 0; rl_state['bars'] = 0
                
        if base_state['pos'] == 0:
            if y_v[0] == 1.0 or y_v[2] == 1.0:
                base_state['pos'] = 1; base_state['entry'] = current_price
            elif y_v[1] == 1.0 or y_v[3] == 1.0 or y_v[4] == 1.0:
                base_state['pos'] = -1; base_state['entry'] = current_price
        else:
            base_state['bars'] += 1
            if (base_state['pos'] == 1 and (y_v[3] == 1.0 or y_v[4] == 1.0)) or \
               (base_state['pos'] == -1 and (y_v[2] == 1.0)):
                r = np.log(current_price / base_state['entry'])
                base_state['trades'] += 1
                if base_state['pos'] == 1:
                    base_state['pnl'] += r
                    if r > 0: base_state['wins'] += 1
                elif base_state['pos'] == -1:
                    base_state['pnl'] -= r
                    if r < 0: base_state['wins'] += 1
                base_state['pos'] = 0; base_state['bars'] = 0

        # Record step-by-step PnL curves
        rl_pnl_history.append(float(rl_state['pnl']))
        base_pnl_history.append(float(base_state['pnl']))

        # Periodically log live metrics
        if i % 100 == 0:
            r_tr = rl_state['trades']; r_pnl = rl_state['pnl']
            r_wr = (rl_state['wins'] / r_tr * 100) if r_tr > 0 else 0
            
            b_tr = base_state['trades']; b_pnl = base_state['pnl']
            b_wr = (base_state['wins'] / b_tr * 100) if b_tr > 0 else 0
            
            print(f"[{current_time}] ")
            print(f"  --> RL Agent PnL: {r_pnl*100:+.2f}% (WR: {r_wr:.1f}% | Trades: {r_tr})")
            print(f"  --> Baseline   PnL: {b_pnl*100:+.2f}% (WR: {b_wr:.1f}% | Trades: {b_tr})")

    # Save the explicitly fine-tuned state
    torch.save(agent.model.state_dict(), 'models/live_adapted_expert_1h.pth')
    print("Live stream completed. Adapted model saved to 'models/live_adapted_expert_1h.pth'")
    r_tr = rl_state['trades']
    wr = (rl_state['wins'] / r_tr) if r_tr > 0 else 0.0
    
    return rl_state['pnl'], base_state['pnl'], r_tr, wr, rl_pnl_history, base_pnl_history, rl_decision_history

if __name__ == "__main__":
    run_live_simulation()
