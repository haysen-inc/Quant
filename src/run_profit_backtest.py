import torch
import numpy as np
import pandas as pd
from src.dataset import get_dataloaders
from src.differentiable_expert import DifferentiableExpertTransformer
from src.data_loader import fetch_spy_data
from src.features_torch import extract_features
from src.labels_torch import extract_labels
from src.train import calculate_financial_metrics

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
    
    # Use interval="1h" to enforce the 35-bar hold period (same as Model 4 training)
    train_loader, val_loader, dataset = get_dataloaders(
        features_dict, labels_dict, df, 
        seq_len=15, batch_size=64, train_split=0.8, interval=interval
    )
    
    model = DifferentiableExpertTransformer().to(device)
    model.load_state_dict(torch.load('models/quant_transformer.pth', map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    all_returns = []
    
    with torch.no_grad():
        for x, y, r in val_loader:
            x, y, r = x.to(device), y.to(device), r.to(device)
            x_dict = {
                'C': x[:, :, 0], 'H': x[:, :, 1], 'L': x[:, :, 2],
                'J1': x[:, :, 11], 'J2': x[:, :, 14], 'J3': x[:, :, 17],
                'JN3_36': x[:, :, 18],
                'JX': x[:, :, 19], 'EMAJX': x[:, :, 20], 'EMAJX8': x[:, :, 21],
                'ma_c_down': x[:, :, 22], 'ma_c_up': x[:, :, 23],
                'JX_base': x[:, :, 24],
                'F1': x[:, :, 25],
                'F2': x[:, :, 26],
                'EMA_JX_base': x[:, :, 27],
                'EMA_F1': x[:, :, 28],
                'EMA_F2': x[:, :, 29]
            }
            logits_seq = model(x_dict)
            probs = logits_seq[:, -1, :]
            
            all_preds.append(probs.cpu())
            all_targets.append(y.cpu())
            all_returns.append(r.cpu())
            
    val_preds = torch.cat(all_preds, dim=0).numpy()
    val_targets = torch.cat(all_targets, dim=0).numpy()[:, :4] # [BK2, SK2, BP1, SP1]
    val_returns = torch.cat(all_returns, dim=0).numpy()
    
    print("\n" + "="*50)
    print("收益导向基准对照实验结果 (Hold Period: 35 hours / ~5 days)")
    print("="*50)
    
    signal_names = ['BK2', 'SK2', 'BP1', 'SP1']
    
    for i in range(4):
        name = signal_names[i]
        
        # 1. 胜率导向：原始人类专家的实际触发 (Where Human Label == 1)
        expert_mask = val_targets[:, i] == 1
        expert_returns = val_returns[expert_mask]
        if i in [1, 3]: # Shorts
            expert_returns = -expert_returns
            
        expert_wins = (expert_returns > 0).sum() if len(expert_returns) > 0 else 0
        expert_win_rate = expert_wins / (len(expert_returns) + 1e-8)
        expert_avg_pnl = expert_returns.mean() if len(expert_returns) > 0 else 0
        expert_total_pnl = expert_returns.sum() if len(expert_returns) > 0 else 0
        
        # 2. 收益导向：Differentiable Expert 模型的 Top K
        # To make it a fair comparison, let's select exactly the same number of trades as the expert!
        trade_count = len(expert_returns)
        if trade_count == 0:
            trade_count = 20 # Fallback 
            
        probs = val_preds[:, i]
        top_indices = np.argsort(probs)[::-1][:trade_count]
        model_returns = val_returns[top_indices]
        if i in [1, 3]: # Shorts
            model_returns = -model_returns
            
        model_wins = (model_returns > 0).sum()
        model_win_rate = model_wins / (trade_count + 1e-8)
        model_avg_pnl = model_returns.mean()
        model_total_pnl = model_returns.sum()
        
        print(f"\n--- 信号: {name} (共对比 {trade_count} 笔交易) ---")
        print(f"【对照组A】专家规则基线 (胜率/规则拟合导向):")
        print(f"   胜率: {expert_win_rate*100:.1f}%, 平均单笔: {expert_avg_pnl*100:.2f}%, 累计总盈亏: {expert_total_pnl*100:.2f}%")
        
        print(f"【对照组B】可微专家模型 (绝对收益 EV 寻优导向):")
        print(f"   胜率: {model_win_rate*100:.1f}%, 平均单笔: {model_avg_pnl*100:.2f}%, 累计总盈亏: {model_total_pnl*100:.2f}%")
        
        improvement = model_total_pnl - expert_total_pnl
        print(f"   >>> 累计总盈亏净提升: {improvement*100:+.2f}%")

if __name__ == "__main__":
    run_profit_backtest()
