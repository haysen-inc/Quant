import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Canonical feature ordering — single source of truth for index mapping.
# train.py MUST reference this instead of hardcoding indices.
FEATURE_KEYS = [
    'C', 'H', 'L',                                                    # 0-2
    'TEU3', 'TEMA3', 'TED', 'TEMA3T1', 'TEMA3T2', 'TEMA3T3',         # 3-8
    'K1', 'D1', 'J1', 'K2', 'D2', 'J2', 'K3', 'D3', 'J3', 'JN3_36', # 9-18
    'JX', 'EMAJX', 'EMAJX8', 'ma_c_down', 'ma_c_up',                 # 19-23
    'JX_base', 'F1', 'F2', 'EMA_JX_base', 'EMA_F1', 'EMA_F2'         # 24-29
]

LABEL_KEYS = ['BK2', 'SK2', 'BP1', 'SP1', 'SP2']

class SPYSequenceDataset(Dataset):
    """
    Rolling window dataset: [T, 30] features → (sequence_length, 30) windows,
    predicting 5 expert labels at the last bar of each window.
    """
    def __init__(self, features_dict: dict, labels_dict: dict, df: pd.DataFrame, sequence_length: int = 15, interval: str = "1d"):
        self.sequence_length = sequence_length
        self.df = df

        # 1. Stack features into [T, 30] using the canonical key order
        tensors = []
        for k in FEATURE_KEYS:
            # Replace NaNs with 0 for neural network stability
            t_clean = torch.nan_to_num(features_dict[k], nan=0.0, posinf=0.0, neginf=0.0)
            tensors.append(t_clean.unsqueeze(1))
            
        self.X_raw = torch.cat(tensors, dim=1) # [T, 30]
        
        # IMPORTANT (Phase 5 Expert Differentiable Architecture):
        # We DO NOT Z-Score Normalize this data! The formulas like `C > ma_c_down` 
        # or `J1 + J2 - 50` rely on the literal mathematical scales of indices and prices.
        self.X = self.X_raw
        
        # 2. Stack the 5 expert labels into (T, 5) Tensor
        l_tensors = []
        for k in LABEL_KEYS:
            l_tensors.append(labels_dict[k].unsqueeze(1))
        self.Y = torch.cat(l_tensors, dim=1) # [T, 5]
        
        # 3. Dynamic Forward Returns (Fluid Holding Periods)
        # Instead of a fixed N-bar hold, calculate continuous returns until the NEXT opposing exit signal fires
        close_prices = torch.tensor(df['Close'].values, dtype=torch.float32)
        r_fwd_dual = torch.zeros((len(close_prices), 2), dtype=torch.float32) # [Long_Return, Short_Return]
        
        sp1 = labels_dict['SP1'].numpy()
        sp2 = labels_dict['SP2'].numpy()
        bp1 = labels_dict['BP1'].numpy()
        
        # Long exits occur when SP1 or SP2 fire
        long_exits = np.where((sp1 == 1.0) | (sp2 == 1.0))[0]
        # Short exits occur when BP1 fires
        short_exits = np.where(bp1 == 1.0)[0]
        
        max_hold = 200 # Failsafe
        
        for t in range(len(close_prices)):
            # Long return: if entered long at bar t, held until next SP1/SP2 exit
            valid_longs = long_exits[long_exits > t]
            idx_long = min(valid_longs[0], t + max_hold) if len(valid_longs) > 0 else t + max_hold
            idx_long = min(idx_long, len(close_prices) - 1)
            r_fwd_dual[t, 0] = (close_prices[idx_long] - close_prices[t]) / close_prices[t]
            
            # Short return: if entered short at bar t, held until next BP1 exit
            valid_shorts = short_exits[short_exits > t]
            idx_short = min(valid_shorts[0], t + max_hold) if len(valid_shorts) > 0 else t + max_hold
            idx_short = min(idx_short, len(close_prices) - 1)
            r_fwd_dual[t, 1] = (close_prices[idx_short] - close_prices[t]) / close_prices[t]
            
        self.R = r_fwd_dual # [T, 2]
        
        # --- FAT TAIL MACRO RETURN TENSOR ---
        # 35-bar unconstrained holding period purely for RL AI evaluation
        r_macro_dual = torch.zeros((len(close_prices), 2), dtype=torch.float32)
        macro_hold = 35
        for t in range(len(close_prices)):
            idx_target = min(t + macro_hold, len(close_prices) - 1)
            r_macro_dual[t, 0] = (close_prices[idx_target] - close_prices[t]) / close_prices[t]
            r_macro_dual[t, 1] = (close_prices[idx_target] - close_prices[t]) / close_prices[t]  # Absolute inversion happens in Loss
        self.R_macro = r_macro_dual # [T, 2]

    def __len__(self):
        # We can only generate windows up to len - sequence_length
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (sequence_length, num_features)
            y: Tensor of shape (5,) representing labels at the LAST day of the sequence
            r: Future return (Baseline logic) at the LAST day of the sequence
            r_macro: Future return (Unconstrained 35-bar logic)
        """
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        x_seq = self.X[start_idx:end_idx]
        
        # The target to predict is the label at the end of the sequence window
        target_idx = end_idx - 1
        y_label = self.Y[target_idx]
        r_fwd = self.R[target_idx]
        r_fwd_macro = self.R_macro[target_idx]
        
        return x_seq, y_label, r_fwd, r_fwd_macro

def get_dataloaders(features, labels, df, seq_len=15, batch_size=32, train_split=0.8, interval="1d"):
    dataset = SPYSequenceDataset(features, labels, df, sequence_length=seq_len, interval=interval)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # We must not shuffle randomly over time! Use Sequential splitting
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset
