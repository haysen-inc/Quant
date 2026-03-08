import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from src.dataset import get_dataloaders
from src.differentiable_expert import DifferentiableExpertTransformer
from src.features_torch import extract_features
from src.labels_torch import extract_labels
from src.data_loader import fetch_spy_data

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    import os
    os.system('pip install scikit-learn')
    from sklearn.metrics import roc_auc_score, average_precision_score

class AsymmetricExpertLoss(nn.Module):
    """
    Multi-Task Loss Function (Loss_Expert + Lambda * Pen_Profit + Gamma * Pen_Cost)
    As agreed upon in Phase 3:
    1. Loss_Expert: Binary Cross Entropy against the 4 expert signals (BK2, SK2, BP1, SP1).
    2. Pen_Profit: Asymmetric Profit Regulator. 
       - If model predicts correctly (y=1) AND it's highly profitable (return > 0): multiply gradient.
       - If it predicts heavily on a losing trade (y=1 AND return <= 0): reduce the confidence target to 0.
    """
    def __init__(self, lambda_profit=50.0, cost_fee=0.0):
        super().__init__()
        self.lambda_profit = lambda_profit
        self.cost_fee = cost_fee
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, probs, targets, forward_returns_dual, prev_probs=None):
        """
        probs: [Batch, 5] -> [BK2, SK2, BP1, SP1, SP2]
        targets: [Batch, 5] binary expert labels
        forward_returns_dual: [Batch, 2] -> [R_long, R_short]
        """
        # 1. Expand dual returns to match the 5 signals
        r_long = forward_returns_dual[:, 0]
        r_short = forward_returns_dual[:, 1]
        
        # BK2(0): Long, SK2(1): Short, BP1(2): Long, SP1(3): Short, SP2(4): Short
        returns_expanded = torch.stack([r_long, r_short, r_long, r_short, r_short], dim=1)
        
        # 2. Invert returns for Short signals (SK2: index 1, SP1: index 3, SP2: index 4)
        adjusted_returns = returns_expanded.clone()
        adjusted_returns[:, [1, 3, 4]] = -adjusted_returns[:, [1, 3, 4]]
        
        # 3. Dynamic Targets & Asymmetric Weights
        dynamic_targets = targets.clone()
        weights = torch.ones_like(probs)
        
        # Case A: Expert fired (y=1), but it LOST money (adjusted_return <= 0).
        # Action: Reduce confidence target to 0, forcing the model to NOT fire here!
        mask_losing_pos = (targets == 1.0) & (adjusted_returns <= 0)
        dynamic_targets[mask_losing_pos] = 0.0
        
        # Case B: ORACLE PROFIT LABELS (The "Zero-to-One" Unleash)
        # We FORCE the Entry signals (BK2:0, SK2:1) to 1.0 if the forward return > 0.2%.
        # We DO NOT force Exit signals (BP1:2, SP1:3, SP2:4) to fire on the same bar! That would cause immediate closing.
        mask_oracle_profit = (adjusted_returns > 0.002)
        
        # Apply Oracle strictly to Entries (Index 0 and 1)
        dynamic_targets[:, 0][mask_oracle_profit[:, 0]] = 1.0
        dynamic_targets[:, 1][mask_oracle_profit[:, 1]] = 1.0
        
        # Apply massive class balancing weights to all 1.0 targets (0.2% sparsity fix)
        weights[dynamic_targets == 1.0] = 50.0
        
        # Case B: Expert fired (y=1), and it MADE money (adjusted_return > 0).
        # Action: Multiply the penalty if it fails to predict it.
        mask_winning_pos = (targets == 1.0) & (adjusted_returns > 0)
        weights[mask_winning_pos] += self.lambda_profit * adjusted_returns[mask_winning_pos]
        
        # 4. Base Expert Loss (Clamped to prevent floating point crashes)
        clamped_probs = torch.clamp(probs, min=1e-6, max=1.0-1e-6)
        clamped_targets = torch.clamp(dynamic_targets, min=0.0, max=1.0)
        base_loss = self.bce(clamped_probs, clamped_targets)
        
        # 5. Apply Asymmetric Weighting
        weighted_loss = (base_loss * weights).mean()
        
        # 6. Transaction Cost Penalty
        pen_cost = torch.tensor(0.0, device=probs.device)
        if prev_probs is not None and probs.shape == prev_probs.shape:
            prob_diff = torch.abs(probs - prev_probs)
            pen_cost = self.cost_fee * prob_diff.mean()
            
        loss_total = weighted_loss + pen_cost
        
        # Also return alpha EV for logging purposes
        buy_ev = (probs[:, [0, 2]] * r_long.unsqueeze(1)).mean()
        sell_ev = (probs[:, [1, 3, 4]] * -r_short.unsqueeze(1)).mean()
        alpha_realized = buy_ev + sell_ev
        
        loss_total = weighted_loss + pen_cost
        
        return loss_total, alpha_realized.item(), pen_cost.item()

def calculate_metrics(y_pred, y_true):
    """
    Calculate PR-AUC and ROC-AUC against human labels just for OBSERVATION.
    The model is no longer trained on these labels.
    y_pred: (Batch, 4) Probabilities
    y_true: (Batch, 5) Binary labels -> We select indices [0, 1, 2, 3] to match `[BK2, SK2, BP1, SP1]` and ignore SP2 for now.
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().numpy() 
    
    pr_aucs = []
    roc_aucs = []
    
    for i in range(5):
        try:
            pr_auc = average_precision_score(y_true[:, i], y_pred[:, i])
            roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            pr_auc = 0.0
            roc_auc = 0.5
        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)
        
    return pr_aucs, roc_aucs

def calculate_financial_metrics(y_pred, y_true, forward_returns_batch, top_k=50):
    y_p = y_pred.cpu().detach().numpy()
    r_fwd = forward_returns_batch.cpu().numpy()
    
    fin_metrics = []
    
    # 0: BK2, 1: SK2, 2: BP1, 3: SP1, 4: SP2
    for i in range(5):
        probs = y_p[:, i]
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # Dual Dimension Return Selector
        if i in [0, 2]: # Long Signal -> Uses SP1/SP2 Exit Array
            selected_returns = r_fwd[top_indices, 0]
        else: # Short Signal -> Uses BP1 Exit Array
            selected_returns = -r_fwd[top_indices, 1]
            
        wins = (selected_returns > 0).sum()
        win_rate = wins / (top_k + 1e-8)
        avg_return = selected_returns.mean()
        
        fin_metrics.append({'win_rate': win_rate, 'avg_return': avg_return})
        
    return fin_metrics

def train_model(epochs=50, batch_size=64, lr=1e-2, seq_len=15, interval="1d", ast_config=None):
    # --- PHASE 14 DETERMINISM FIX ---
    # Lock all RNG seeds to guarantee identical converged profitability and stop the lottery
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    import datetime
    today = datetime.datetime.now()
    
    start_date = "2020-01-01"
    if interval == "1h":
        start_date = (today - datetime.timedelta(days=729)).strftime('%Y-%m-%d')
    elif interval == "15m":
        start_date = (today - datetime.timedelta(days=59)).strftime('%Y-%m-%d')
        
    print(f"Loading data ({interval}) and calculating expert initial tensors...")
    df = fetch_spy_data(start_date=start_date, end_date=None, interval=interval)
    
    if df is None or len(df) == 0:
        return
        
    features_dict = extract_features(df)
    labels_dict = extract_labels(df, features_dict) # Still extracted for printing validation tests
    
    train_loader, val_loader, dataset = get_dataloaders(
        features_dict, labels_dict, df, 
        seq_len=seq_len, batch_size=batch_size, train_split=0.8, interval=interval
    )
    
    # 3. Initialize Differentiable Expert with Dynamic AST payload
    model = DifferentiableExpertTransformer(init_constants=ast_config).to(device)
    
    # Phase 9/18: Finding Escape Velocity LR
    # We increase the learning rate from 0.05 up to 0.25 for expert parameters.
    # The Transformer branch parameters will inherit the standard optimizer learning rate.
    expert_constants = ['expert_prior.w_bias', 'expert_prior.w_f1', 'expert_prior.w_f2', 'expert_prior.w_cond3_j1']
    temperatures = [n for n, p in model.named_parameters() if 'temp' in n]
    high_lr_names = expert_constants + temperatures
    
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if n in high_lr_names], 'lr': 0.25},
        {'params': [p for n, p in model.named_parameters() if n not in high_lr_names], 'lr': lr}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)    
    criterion = AsymmetricExpertLoss()
    
    writer = SummaryWriter(log_dir=f'logs/diff_expert_{interval}')
    
    history = {
        'train_loss': [], 'val_loss': [], 'best_pnl': 0.0, 'best_win_rate': 0.0,
        'w_bias_history': [], 'w_f1_history': [], 'w_f2_history': [], 'w_cond3_j1_history': [],
        'temp_dg_jx_rx': [], 'temp_dg_jx_j1': [], 'temp_dg_c_macd': [], 
        'temp_dl_jx_rx': [], 'temp_dl_c_macu': []
    }
    
    # --- PHASE 12: Early Stopping & Best Checkpoint Tracking ---
    best_pnl = -float('inf')
    best_model_path = f'models/best_expert_{interval}.pth'
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_alpha = 0
        
        prev_probs = None 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y, r) in enumerate(pbar):
            # We unpack the [Batch, Seq, 23] tensor back into a dict for the model
            # Based on features_torch.py order: 
            # 0:C, 1:H, 2:L, ..., 11:J1, 14:J2, 17:J3, 18:JN3_36, 19:JX, 20:EMAJX, 21:EMAJX8, 22:ma_c_down, 23:ma_c_up
            x_dict = {
                'C': x[:, :, 0],
                'H': x[:, :, 1],
                'L': x[:, :, 2],
                'J1': x[:, :, 11],
                'J2': x[:, :, 14],
                'J3': x[:, :, 17],
                'JN3_36': x[:, :, 18],
                'JX': x[:, :, 19],
                'EMAJX': x[:, :, 20],
                'EMAJX8': x[:, :, 21],
                'ma_c_down': x[:, :, 22],
                'ma_c_up': x[:, :, 23],
                'JX_base': x[:, :, 24],
                'F1': x[:, :, 25],
                'F2': x[:, :, 26],
                'EMA_JX_base': x[:, :, 27],
                'EMA_F1': x[:, :, 28],
                'EMA_F2': x[:, :, 29]
            }
            
            # Forward pass: Output is [Batch, Seq, 5]
            # We predict using the last day's sequence state
            logits_seq = model(x_dict)
            probs = logits_seq[:, -1, :] # [Batch, 5] -> [BK2, SK2, BP1, SP1, SP2]
            
            # Pure Alpha Loss
            loss, alpha_realized, l_cost = criterion(probs, y.float(), r, prev_probs=prev_probs)
            
            if not loss.requires_grad:
                loss.requires_grad_(True)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_alpha += alpha_realized
            prev_probs = probs.detach()
            
            pbar.set_postfix({'EV_Loss': f"{loss.item():.4f}", 'Real_Alpha': f"{alpha_realized:.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_alpha = epoch_alpha / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        all_returns = []
        
        with torch.no_grad():
            for x, y, r in val_loader:
                x, y, r = x.to(device), y.to(device), r.to(device)
                
                x_dict = {
                    'C': x[:, :, 0],
                    'H': x[:, :, 1],
                    'L': x[:, :, 2],
                    'J1': x[:, :, 11],
                    'J2': x[:, :, 14],
                    'J3': x[:, :, 17],
                    'JN3_36': x[:, :, 18],
                    'JX': x[:, :, 19],
                    'EMAJX': x[:, :, 20],
                    'EMAJX8': x[:, :, 21],
                    'ma_c_down': x[:, :, 22],
                    'ma_c_up': x[:, :, 23],
                    'JX_base': x[:, :, 24],
                    'F1': x[:, :, 25],
                    'F2': x[:, :, 26],
                    'EMA_JX_base': x[:, :, 27],
                    'EMA_F1': x[:, :, 28],
                    'EMA_F2': x[:, :, 29]
                }
                
                logits_seq = model(x_dict)
                probs = logits_seq[:, -1, :]
                
                loss, _, _ = criterion(probs, y.float(), r)
                val_loss += loss.item()
                
                all_preds.append(probs)
                all_targets.append(y)
                all_returns.append(r)
                
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Metrics over all validation data
        val_preds = torch.cat(all_preds, dim=0)
        val_targets = torch.cat(all_targets, dim=0) # [Batch, 5] true
        val_returns = torch.cat(all_returns, dim=0)
        
        pr_aucs, roc_aucs = calculate_metrics(val_preds, val_targets)
        fin_metrics = calculate_financial_metrics(val_preds, val_targets, val_returns, top_k=20)
        
        print(f"Epoch {epoch+1}/{epochs} - Train EV_Loss: {avg_train_loss:.4f} (Alpha: {avg_train_alpha:.4f}) - Val EV_Loss: {avg_val_loss:.4f}")
        # Print DL metrics for BP1 (Index 2 in the 4-dim output)
        print(f"   [BP1 Tracking] Expert Mimic ROC-AUC: {roc_aucs[2]:.4f}")
        bp1_fin = fin_metrics[2]
        print(f"   [BP1/SP1 Finance] BP1 Win Rate: {bp1_fin['win_rate']*100:.1f}%, BP1 Avg PnL: {bp1_fin['avg_return']*100:.2f}% | SP1 Win Rate: {fin_metrics[3]['win_rate']*100:.1f}%, SP1 Avg PnL: {fin_metrics[3]['avg_return']*100:.2f}%")

        # --- PHASE 12: Best Model Checkpointing Logic ---
        # We track the absolute PnL generated by the highly profitable True Entry Signals (BK2:0, SK2:1)
        long_pnl = fin_metrics[0]['avg_return']
        short_pnl = fin_metrics[1]['avg_return']
        current_pnl = (long_pnl + short_pnl) / 2.0
        
        current_win_rate = (fin_metrics[0]['win_rate'] + fin_metrics[1]['win_rate']) / 2.0
        
        if current_pnl > best_pnl:
            best_pnl = current_pnl
            history['best_pnl'] = float(current_pnl)
            history['best_win_rate'] = float(current_win_rate)
            torch.save(model.state_dict(), best_model_path)
            print(f"   ---> [NEW HIGH] Model checkpoint saved! Combined Entry Signals Avg PnL: {current_pnl*100:.2f}%")
        
        # --- TensorBoard Logging ---
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        signal_names = ['BK2', 'SK2', 'BP1', 'SP1']
        for i, name in enumerate(signal_names):
            writer.add_scalar(f'Metrics_PR_AUC/{name}', pr_aucs[i], epoch)
            writer.add_scalar(f'Metrics_ROC_AUC/{name}', roc_aucs[i], epoch)
            writer.add_scalar(f'Finance_WinRate/{name}', fin_metrics[i]['win_rate'], epoch)
            writer.add_scalar(f'Finance_AvgPnL/{name}', fin_metrics[i]['avg_return'], epoch)
            
        # Log the evolution of the newly opened Expert Parameter Space (-50, 6, 6)
        w_b = model.expert_prior.w_bias.item()
        w_f1 = model.expert_prior.w_f1.item()
        w_f2 = model.expert_prior.w_f2.item()
        w_c = model.expert_prior.w_cond3_j1.item()
        
        writer.add_scalar('Expert_Constants/w_bias', w_b, epoch)
        writer.add_scalar('Expert_Constants/w_f1', w_f1, epoch)
        writer.add_scalar('Expert_Constants/w_f2', w_f2, epoch)
        
        history['w_bias_history'].append(w_b)
        history['w_f1_history'].append(w_f1)
        history['w_f2_history'].append(w_f2)
        history['w_cond3_j1_history'].append(w_c)
        
        history['temp_dg_jx_rx'].append(model.expert_prior.dg_jx_rx.temperature.item())
        history['temp_dg_jx_j1'].append(model.expert_prior.dg_jx_j1.temperature.item())
        history['temp_dg_c_macd'].append(model.expert_prior.dg_c_macd.temperature.item())
        history['temp_dl_jx_rx'].append(model.expert_prior.dl_jx_rx.temperature.item())
        history['temp_dl_c_macu'].append(model.expert_prior.dl_c_macu.temperature.item())
        
        # Log other logic parameters like temperatures if they exist
        for name, param in model.named_parameters():
            if 'temp' in name or 'w_cond' in name:
                writer.add_histogram(f'Expert_Gates/{name}', param.clone().cpu().data.numpy(), epoch)
                    
    writer.close()
        
    # Phase 12 Finality: Load the parameters from the absolute Best Profit epoch instead of the final noisy epoch
    if os.path.exists(best_model_path):
        print(f"\nTraining Complete. Rolling back to best PnL snapshot (Peak Average Core PnL: {best_pnl*100:.2f}%)")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        # Keep the final model name compatible with the backtesting scripts
        torch.save(model.state_dict(), 'models/quant_transformer.pth')
    else:
        # Fallback to saving the last epoch
        torch.save(model.state_dict(), 'models/quant_transformer.pth')
        
    print(f"Final Model secured at 'models/quant_transformer.pth'")
    return history

if __name__ == "__main__":
    # 正式训练运行 (Formal Training Run)
    # 使用 1h 级别数据进行 50 轮深层特征抽取训练
    print("=" * 50)
    print("🚀 启动 SPY 专家先验 Transformer 量化模型正式训练")
    print("=" * 50)
    history = train_model(epochs=50, batch_size=128, interval="1h")
