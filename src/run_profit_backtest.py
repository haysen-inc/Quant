import torch
import numpy as np
import pandas as pd
from src.dataset import get_dataloaders, FEATURE_KEYS
from src.differentiable_expert import DifferentiableExpertTransformer
from src.data_loader import fetch_spy_data
from src.features_torch import extract_features
from src.labels_torch import extract_labels
from src.train import calculate_financial_metrics, _build_x_dict

def run_profit_backtest():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data for strict backtest...")

    import datetime
    today = datetime.datetime.now()
    start_date = (today - datetime.timedelta(days=729)).strftime('%Y-%m-%d')
    interval = "1h"

    df = fetch_spy_data(start_date=start_date, end_date=None, interval=interval)
    features_dict = extract_features(df)
    labels_dict = extract_labels(df, features_dict)

    train_loader, val_loader, dataset = get_dataloaders(
        features_dict, labels_dict, df,
        seq_len=15, batch_size=64, train_split=0.8, interval=interval
    )

    model = DifferentiableExpertTransformer().to(device)
    state_dict = torch.load('models/quant_transformer.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_preds = []
    all_targets = []
    all_returns = []

    with torch.no_grad():
        for x, y, r, r_macro in val_loader:
            x, y, r_macro = x.to(device), y.to(device), r_macro.to(device)
            x_dict = _build_x_dict(x)
            logits_seq = model(x_dict)
            probs = logits_seq[:, -1, :]

            all_preds.append(probs.cpu())
            all_targets.append(y.cpu())
            all_returns.append(r_macro.cpu())

    val_preds = torch.cat(all_preds, dim=0).numpy()
    val_targets = torch.cat(all_targets, dim=0).numpy()  # [BK2, SK2, BP1, SP1, SP2] all 5
    val_returns = torch.cat(all_returns, dim=0).numpy()

    print("\n" + "="*50)
    print("收益导向基准对照实验结果 (Hold Period: 35 hours / ~5 days)")
    print("="*50)

    # Long-only: BK2=开多, SK2/SP1/SP2=平多, BP1=irrelevant
    # Only evaluate BK2 (entry) and exit signals with r_long
    signal_names = ['BK2', 'SK2', 'SP1', 'SP2']
    signal_indices = [0, 1, 3, 4]

    for idx, i in enumerate(signal_indices):
        name = signal_names[idx]

        # 1. Expert baseline: edge-detected signals only (0→1 transitions)
        raw_labels = val_targets[:, i]
        edge_mask = np.zeros(len(raw_labels), dtype=bool)
        for t in range(1, len(raw_labels)):
            if raw_labels[t] == 1 and raw_labels[t-1] == 0:
                edge_mask[t] = True
        if raw_labels[0] == 1:
            edge_mask[0] = True

        # Long-only: all signals use r_long (column 0)
        if i == 0:      # BK2: Open Long → r_long
            expert_returns = val_returns[edge_mask, 0]
        else:           # SK2/SP1/SP2: Close Long → -r_long
            expert_returns = -val_returns[edge_mask, 0]

        expert_wins = (expert_returns > 0).sum() if len(expert_returns) > 0 else 0
        expert_win_rate = expert_wins / (len(expert_returns) + 1e-8)
        expert_avg_pnl = expert_returns.mean() if len(expert_returns) > 0 else 0
        expert_total_pnl = expert_returns.sum() if len(expert_returns) > 0 else 0

        # 2. Model: Top K by probability
        trade_count = int(edge_mask.sum())
        if trade_count == 0:
            trade_count = 20

        probs = val_preds[:, i]
        top_indices = np.argsort(probs)[::-1][:trade_count]

        if i == 0:
            model_returns = val_returns[top_indices, 0]
        else:
            model_returns = -val_returns[top_indices, 0]

        model_wins = (model_returns > 0).sum()
        model_win_rate = model_wins / (trade_count + 1e-8)
        model_avg_pnl = model_returns.mean()
        model_total_pnl = model_returns.sum()

        print(f"\n--- 信号: {name} (共对比 {trade_count} 笔交易, edge-detected) ---")
        print(f"【对照组A】专家规则基线:")
        print(f"   胜率: {expert_win_rate*100:.1f}%, 平均单笔: {expert_avg_pnl*100:.2f}%, 累计总盈亏: {expert_total_pnl*100:.2f}%")

        print(f"【对照组B】可微专家模型:")
        print(f"   胜率: {model_win_rate*100:.1f}%, 平均单笔: {model_avg_pnl*100:.2f}%, 累计总盈亏: {model_total_pnl*100:.2f}%")

        improvement = model_total_pnl - expert_total_pnl
        print(f"   >>> 累计总盈亏净提升: {improvement*100:+.2f}%")

if __name__ == "__main__":
    run_profit_backtest()
