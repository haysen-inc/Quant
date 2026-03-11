import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from src.dataset import get_dataloaders, FEATURE_KEYS
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
    1. Loss_Expert: Binary Cross Entropy against the 5 expert signals.
    2. Pen_Profit: Asymmetric Profit Regulator.
       - If model predicts correctly (y=1) AND it's highly profitable (return > 0): multiply gradient.
       - If it predicts heavily on a losing trade (y=1 AND return <= 0): reduce the confidence target to 0.
    """
    def __init__(self, lambda_profit=50.0, cost_fee=0.0):
        super().__init__()
        self.lambda_profit = lambda_profit
        self.cost_fee = cost_fee
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, probs, targets, forward_returns_dual, prev_probs=None, use_bce=True):
        """
        probs: [Batch, 5] -> [BK2, SK2, BP1, SP1, SP2]
        targets: [Batch, 5] binary expert labels
        forward_returns_dual: [Batch, 2] -> [R_long, R_short]
        """
        # LONG-ONLY strategy: all signals relate to r_long only
        # BK2(0): 开多 (Open Long)  → r_long, fire when price going UP
        # SK2(1): 平多 (Close Long) → r_long, fire when price going DOWN (exit to save profit)
        # BP1(2): irrelevant in long-only, low weight
        # SP1(3): 平多 (Close Long) → r_long, fire when price going DOWN
        # SP2(4): 平多 (Close Long) → r_long, fire when price going DOWN
        r_long = forward_returns_dual[:, 0]

        # adjusted_returns: positive = "this signal should fire"
        # BK2:  +r_long (fire when up → go long)
        # SK2:  -r_long (fire when down → exit long)
        # BP1:  0 (irrelevant in long-only)
        # SP1:  -r_long (fire when down → exit long)
        # SP2:  -r_long (fire when down → exit long)
        adjusted_returns = torch.stack([
            r_long, -r_long, torch.zeros_like(r_long), -r_long, -r_long
        ], dim=1)

        # Signal importance weights: BP1 gets near-zero weight (irrelevant for long-only)
        signal_weights = torch.tensor([1.0, 1.0, 0.05, 1.0, 1.0], device=probs.device)

        # 3. Dynamic Targets & Asymmetric Weights
        dynamic_targets = targets.clone()
        weights = torch.ones_like(probs) * signal_weights.unsqueeze(0)

        # Case A: Expert fired (y=1), but it LOST money (adjusted_return <= 0).
        mask_losing_pos = (targets == 1.0) & (adjusted_returns <= 0)
        # BK2 entry signals: use soft target 0.3 to preserve entry frequency
        # Exit signals: hard suppress to 0.0
        bk2_losing = mask_losing_pos[:, 0]
        exit_losing = mask_losing_pos.clone()
        exit_losing[:, 0] = False
        dynamic_targets[exit_losing] = 0.0
        dynamic_targets[bk2_losing, 0] = 0.3

        # Class balancing weights for sparse 1.0 targets
        weights[dynamic_targets == 1.0] *= 50.0

        # Case C: Expert fired (y=1), and it MADE money — amplify gradient
        mask_winning_pos = (targets == 1.0) & (adjusted_returns > 0)
        weights[mask_winning_pos] += self.lambda_profit * adjusted_returns[mask_winning_pos]

        # 4. Base Expert Loss
        clamped_probs = torch.clamp(probs, min=1e-6, max=1.0-1e-6)
        clamped_targets = torch.clamp(dynamic_targets, min=0.0, max=1.0)

        # BCE component (always computed for structure preservation)
        base_loss = self.bce(clamped_probs, clamped_targets)
        bce_loss = (base_loss * weights).mean()

        if use_bce:
            weighted_loss = bce_loss
        else:
            # Mixed loss: 30% EV + 70% BCE — conservative ratio to prevent overfitting
            # BCE preserves expert structure; EV provides controlled divergence
            scaled_adjusted = adjusted_returns * 100.0
            weighted_ev = probs * scaled_adjusted * signal_weights.unsqueeze(0)
            ev_loss = -self.lambda_profit * weighted_ev.sum(dim=-1).mean()
            weighted_loss = 0.3 * ev_loss + 0.7 * bce_loss

        # 5. Mutual exclusion penalty: BK2 and exit signals should not both be high
        p_entry = probs[:, 0]  # BK2
        p_exit_max = torch.max(probs[:, [1, 3, 4]], dim=1).values  # max(SK2, SP1, SP2)
        conflict = torch.min(p_entry, p_exit_max)
        pen_conflict = 50.0 * conflict.mean()

        # Sparsity: penalize signals whose batch-mean exceeds cap (prevents saturation)
        # BK2 uses higher cap (0.4) since expert fires at 31% — 0.3 cap suppresses it
        signal_means = probs.mean(dim=0)  # [5]
        sparsity_caps = torch.tensor([0.40, 0.30, 0.30, 0.30, 0.30], device=probs.device)
        pen_sparsity = 20.0 * torch.clamp(signal_means - sparsity_caps, min=0.0).sum()

        # Anti-collapse: BK2 entry needs higher floor (20%) to prevent over-suppression
        # Exit signals keep 5% floor
        pen_dead = 20.0 * (
            torch.clamp(0.20 - signal_means[0], min=0.0) +  # BK2 entry: floor 20%
            torch.clamp(0.05 - signal_means[1], min=0.0) +  # SK2 exit
            torch.clamp(0.05 - signal_means[3], min=0.0) +  # SP1 exit
            torch.clamp(0.05 - signal_means[4], min=0.0)    # SP2 exit
        )

        # 6. Transaction Cost Penalty
        pen_cost = torch.tensor(0.0, device=probs.device)
        if prev_probs is not None and probs.shape == prev_probs.shape:
            prob_diff = torch.abs(probs - prev_probs)
            pen_cost = self.cost_fee * prob_diff.mean()

        # Alpha EV for logging
        alpha_realized = (probs * adjusted_returns * signal_weights.unsqueeze(0)).mean()

        loss_total = weighted_loss + pen_cost + pen_conflict + pen_sparsity + pen_dead

        return loss_total, alpha_realized.item(), pen_cost.item()

def calculate_metrics(y_pred, y_true):
    """Calculate PR-AUC and ROC-AUC for all 5 signals."""
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
    y_t = y_true.cpu().numpy()
    r_fwd = forward_returns_batch.cpu().numpy()

    fin_metrics = []

    for i in range(5):
        probs = y_p[:, i]

        top_indices = np.where(probs >= 0.5)[0]
        trade_count = len(top_indices)

        if trade_count == 0:
            trade_count = 1
            top_indices = np.argsort(probs)[::-1][:1]

        # Long-only: all signals use r_long (column 0)
        if i == 0:      # BK2: Open Long
            selected_returns = r_fwd[top_indices, 0]
        elif i == 2:    # BP1: irrelevant, use 0
            selected_returns = np.zeros(len(top_indices))
        else:           # SK2(1), SP1(3), SP2(4): Close Long
            selected_returns = -r_fwd[top_indices, 0]

        wins = (selected_returns > 0).sum()
        win_rate = wins / (trade_count + 1e-8)
        avg_return = selected_returns.mean()
        sum_return = selected_returns.sum()

        fin_metrics.append({'win_rate': win_rate, 'avg_return': avg_return, 'sum_return': sum_return, 'trades': trade_count})

    return fin_metrics


def _build_x_dict(x_batch, device=None):
    """Unpack stacked [Batch, Seq, 30] tensor into x_dict using canonical FEATURE_KEYS ordering."""
    if device is not None:
        x_batch = x_batch.to(device)
    return {key: x_batch[:, :, i] for i, key in enumerate(FEATURE_KEYS)}


def train_model(epochs=50, lr=1e-2, seq_len=15, interval="1h", ast_config=None,
                expert_lr=0.25, feature_lr=0.01, **kwargs):
    """
    Unified end-to-end training: raw OHLC → learnable features → expert → transformer.

    Full-series forward pass per epoch. All parameters (20 feature alphas + 4 expert
    constants + 7 gate temperatures + transformer weights) train end-to-end.

    Parameter groups with separate learning rates:
      - Feature extractor (20 DiffEMA alphas): feature_lr (conservative, avoid destabilizing indicators)
      - Expert prior (4 constants + 7 temps): expert_lr (moderate, tune strategy logic)
      - Transformer (attention + projection): lr (standard, learn residual corrections)
    """
    import random
    import datetime

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    today = datetime.datetime.now()
    start_date = "2020-01-01"
    if interval == "1h":
        start_date = (today - datetime.timedelta(days=729)).strftime('%Y-%m-%d')
    elif interval == "15m":
        start_date = (today - datetime.timedelta(days=59)).strftime('%Y-%m-%d')

    print(f"Loading data ({interval})...")
    df = fetch_spy_data(start_date=start_date, end_date=None, interval=interval)
    if df is None or len(df) == 0:
        return {}

    # Labels from FIXED features (expert ground truth — never changes during training)
    fixed_features = extract_features(df)
    labels_dict = extract_labels(df, fixed_features, ast_config=ast_config)

    from src.dataset import LABEL_KEYS

    # Raw OHLC tensors for end-to-end forward pass
    C = torch.tensor(df['Close'].values, dtype=torch.float32).to(device)
    H = torch.tensor(df['High'].values, dtype=torch.float32).to(device)
    L = torch.tensor(df['Low'].values, dtype=torch.float32).to(device)

    # Stacked labels [T, 5]
    Y = torch.stack([torch.nan_to_num(labels_dict[k], nan=0.0) for k in LABEL_KEYS], dim=1).to(device)

    # Forward returns (35-bar macro hold) — vectorized
    T = len(C)
    close_det = C.detach()
    future_idx = (torch.arange(T, device=device) + 35).clamp(max=T - 1)
    r = (close_det[future_idx] - close_det) / close_det
    R_macro = torch.stack([r, r], dim=1)  # [T, 2]

    # Train/val split (skip indicator warm-up bars)
    # 60/40 split: larger val set covers more market regimes → better generalization signal
    warm_up = 210
    total_valid = T - warm_up
    train_end = warm_up + int(total_valid * 0.6)

    print(f"T={T}, warm_up={warm_up}, train=[{warm_up}:{train_end}], val=[{train_end}:{T}]")

    model = DifferentiableExpertTransformer(init_constants=ast_config).to(device)

    # 3 parameter groups with separate learning rates
    feat_ids = set(id(p) for p in model.feature_extractor.parameters())
    expert_ids = set(id(p) for p in model.expert_prior.parameters())

    param_groups = [
        {'params': list(model.feature_extractor.parameters()), 'lr': feature_lr},
        {'params': list(model.expert_prior.parameters()), 'lr': expert_lr},
        {'params': [p for n, p in model.named_parameters()
                    if id(p) not in feat_ids and id(p) not in expert_ids], 'lr': lr}
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = AsymmetricExpertLoss()

    writer = SummaryWriter(log_dir=f'logs/diff_expert_{interval}')

    history = {
        'train_loss': [], 'val_loss': [], 'best_pnl': 0.0, 'best_win_rate': 0.0,
        'best_base_pnl': 0.0, 'best_base_win_rate': 0.0,
        'w_bias_history': [], 'w_f1_history': [], 'w_f2_history': [], 'w_cond3_j1_history': [],
        'temp_dg_jx_rx': [], 'temp_dg_jx_j1': [], 'temp_dg_c_macd': [],
        'temp_dl_jx_rx': [], 'temp_dl_c_macu': []
    }

    best_pnl = -float('inf')
    best_model_path = f'models/best_expert_{interval}.pth'
    os.makedirs('models', exist_ok=True)
    patience = 15
    patience_counter = 0

    from evaluate_baseline import long_only_state_machine

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        optimizer.zero_grad()

        # Full-series end-to-end forward: raw OHLC → features → expert → transformer
        all_probs, expert_probs = model(C, H, L, seq_len=seq_len)

        # Loss on training portion only
        train_probs = all_probs[warm_up:train_end]
        train_y = Y[warm_up:train_end]
        train_r = R_macro[warm_up:train_end]

        use_bce = (epoch < 10)
        train_loss, train_alpha, _ = criterion(train_probs, train_y, train_r, use_bce=use_bce)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history['train_loss'].append(train_loss.item())

        # --- Validation (clean eval mode, no dropout) ---
        model.eval()
        with torch.no_grad():
            all_probs_eval, _ = model(C, H, L, seq_len=seq_len)

            val_probs = all_probs_eval[train_end:]
            val_y = Y[train_end:]
            val_r = R_macro[train_end:]
            val_loss, _, _ = criterion(val_probs, val_y, val_r, use_bce=use_bce)
            history['val_loss'].append(val_loss.item())

            # State machine checkpoint
            vp_np = val_probs.cpu().numpy()
            vy_np = val_y.cpu().numpy()
            val_closes = df['Close'].values[train_end:train_end + len(vy_np)]

            model_hard_signals = (vp_np >= 0.5).astype(float)
            model_state = long_only_state_machine(model_hard_signals, val_closes, start_idx=0)
            model_trades = model_state['trades']
            current_win_rate = model_state['wins'] / max(1, model_trades)
            min_trades = 10
            trade_scale = min(1.0, model_trades / min_trades) if model_trades > 0 else 0.0
            current_pnl = model_state['pnl'] * trade_scale

            base_state = long_only_state_machine(vy_np, val_closes, start_idx=0)
            expert_pnl = base_state['pnl']
            expert_wr = base_state['wins'] / max(1, base_state['trades'])

        # --- Log parameters ---
        w_b = model.expert_prior.w_bias.item()
        w_f1 = model.expert_prior.w_f1.item()
        w_f2 = model.expert_prior.w_f2.item()
        w_c = model.expert_prior.w_cond3_j1.item()

        history['w_bias_history'].append(w_b)
        history['w_f1_history'].append(w_f1)
        history['w_f2_history'].append(w_f2)
        history['w_cond3_j1_history'].append(w_c)
        history['temp_dg_jx_rx'].append(model.expert_prior.dg_jx_rx.temperature.item())
        history['temp_dg_jx_j1'].append(model.expert_prior.dg_jx_j1.temperature.item())
        history['temp_dg_c_macd'].append(model.expert_prior.dg_c_macd.temperature.item())
        history['temp_dl_jx_rx'].append(model.expert_prior.dl_jx_rx.temperature.item())
        history['temp_dl_c_macu'].append(model.expert_prior.dl_c_macu.temperature.item())

        # TensorBoard
        writer.add_scalar('Loss/Train', train_loss.item(), epoch)
        writer.add_scalar('Loss/Validation', val_loss.item(), epoch)
        writer.add_scalar('Expert_Constants/w_bias', w_b, epoch)
        writer.add_scalar('Expert_Constants/w_f1', w_f1, epoch)
        writer.add_scalar('Expert_Constants/w_f2', w_f2, epoch)
        writer.add_scalar('Expert_Constants/w_cond3_j1', w_c, epoch)

        for name, param in model.feature_extractor.named_parameters():
            effective_alpha = torch.sigmoid(param).item()
            writer.add_scalar(f'Feature_Alpha/{name}', effective_alpha, epoch)

        for name, param in model.named_parameters():
            if 'temp' in name and param.numel() == 1:
                writer.add_scalar(f'Expert_Gates/{name}', param.item(), epoch)

        # --- Full-data evaluation (expert tracking metric) ---
        with torch.no_grad():
            full_probs_np = all_probs_eval[warm_up:].cpu().numpy()
            full_labels_np = Y[warm_up:].cpu().numpy()
            full_closes = df['Close'].values[warm_up:warm_up + len(full_probs_np)]

            full_model_hard = (full_probs_np >= 0.5).astype(float)
            full_model_state = long_only_state_machine(full_model_hard, full_closes, start_idx=0)
            full_expert_state = long_only_state_machine(full_labels_np, full_closes, start_idx=0)

            full_model_pnl = full_model_state['pnl']
            full_expert_pnl = full_expert_state['pnl']
            full_model_trades = full_model_state['trades']
            full_model_wr = full_model_state['wins'] / max(1, full_model_trades)

        # --- Print epoch summary ---
        mode_str = "BCE" if use_bce else "EV+BCE"
        print(f"Epoch {epoch+1}/{epochs} ({mode_str}) - Train: {train_loss.item():.4f} "
              f"Val: {val_loss.item():.4f} (Alpha: {train_alpha:.4f})")
        print(f"   Val  SM: {current_pnl*100:+.2f}% ({model_trades}t, WR:{current_win_rate*100:.1f}%) | "
              f"Expert: {expert_pnl*100:+.2f}% ({base_state['trades']}t, WR:{expert_wr*100:.1f}%)")
        print(f"   Full SM: {full_model_pnl*100:+.2f}% ({full_model_trades}t, WR:{full_model_wr*100:.1f}%) | "
              f"Expert: {full_expert_pnl*100:+.2f}% ({full_expert_state['trades']}t)")

        # Checkpoint: use VALIDATION-ONLY PnL as metric
        # Model must prove itself on unseen data to be saved
        if current_pnl > best_pnl:
            best_pnl = current_pnl
            patience_counter = 0
            history['best_pnl'] = float(full_model_pnl)
            history['best_win_rate'] = float(full_model_wr)
            history['best_base_pnl'] = float(full_expert_pnl)
            history['best_base_win_rate'] = float(full_expert_state['wins'] / max(1, full_expert_state['trades']))
            torch.save(model.state_dict(), best_model_path)
            print(f"   ---> [NEW HIGH] Val PnL={current_pnl*100:+.2f}% checkpoint saved!")
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= patience and epoch >= 20:
            print(f"   Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    writer.close()

    # Load best checkpoint
    if os.path.exists(best_model_path):
        print(f"\nTraining Complete. Rolling back to best PnL snapshot (Peak PnL: {best_pnl*100:+.2f}%)")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        torch.save(model.state_dict(), 'models/quant_transformer.pth')
    else:
        torch.save(model.state_dict(), 'models/quant_transformer.pth')

    print(f"Final Model secured at 'models/quant_transformer.pth'")

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Model PnL: {history['best_pnl']*100:.2f}%")
    print(f"Best Model WR:  {history['best_win_rate']*100:.1f}%")
    print(f"Expert Base PnL: {history['best_base_pnl']*100:.2f}%")
    print(f"Expert Base WR:  {history['best_base_win_rate']*100:.1f}%")

    # Print learned feature alphas (EMA/SMA effective periods)
    print("\nLearned Feature Parameters (EMA/SMA effective periods):")
    for name, param in model.feature_extractor.named_parameters():
        alpha = torch.sigmoid(param).item()
        if 'sma' in name:
            eff_period = 1.0 / alpha
        else:
            eff_period = 2.0 / alpha - 1.0
        print(f"  {name}: alpha={alpha:.4f} (effective period ~ {eff_period:.1f})")

    print(f"\nExpert Constants: w_bias={w_b:.4f} w_f1={w_f1:.4f} w_f2={w_f2:.4f} w_cond3={w_c:.4f}")

    return history


if __name__ == "__main__":
    print("=" * 50)
    print("Starting SPY Differentiable Expert Transformer Training")
    print("=" * 50)
    history = train_model(epochs=50, interval="1h")
