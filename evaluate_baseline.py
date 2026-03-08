import pandas as pd
import numpy as np
import torch
import os
from src.data_loader import fetch_spy_data
from src.features_torch import extract_features
from src.labels_torch import extract_labels
import datetime

def evaluate():
    from src.data_loader import fetch_spy_data
    # Use full historical data
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=700) # 700 to be safe from 730 limit
    df = fetch_spy_data(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), interval="1h")
    
    features_dict = extract_features(df)
    labels_dict = extract_labels(df, features_dict)
    
    base_state = {'pos': 0, 'entry': 0.0, 'bars': 0, 'pnl': 0.0, 'wins': 0, 'trades': 0}
    
    y_bk2 = labels_dict['BK2'].numpy()
    y_sk2 = labels_dict['SK2'].numpy()
    y_bp1 = labels_dict['BP1'].numpy()
    y_sp1 = labels_dict['SP1'].numpy()
    y_sp2 = labels_dict['SP2'].numpy()
    
    closes = df['Close'].values
    
    # Simulate exactly like pending queue, but for human expert strict tracker
    for i in range(15, len(df)):
        current_price = closes[i]
        
        y_v = [y_bk2[i], y_sk2[i], y_bp1[i], y_sp1[i], y_sp2[i]]
        
        if base_state['pos'] == 0:
            if y_v[0] == 1.0 or y_v[2] == 1.0:
                base_state['pos'] = 1
                base_state['entry'] = current_price
            elif y_v[1] == 1.0 or y_v[3] == 1.0 or y_v[4] == 1.0:
                base_state['pos'] = -1
                base_state['entry'] = current_price
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
                base_state['pos'] = 0
                base_state['bars'] = 0

    wr = (base_state['wins'] / base_state['trades']) * 100 if base_state['trades'] > 0 else 0
    print(f"--- Full Historical Expert Baseline (Last 2 Years) ---")
    print(f"Total Rows Evaluated: {len(df)}")
    print(f"Total Trades: {base_state['trades']}")
    print(f"Win Rate: {wr:.2f}%")
    print(f"Cumulative PnL: {base_state['pnl']*100:.2f}%")
    if base_state['trades'] > 0:
        print(f"Average PnL per trade: {(base_state['pnl']/base_state['trades'])*100:.4f}%")

if __name__ == "__main__":
    evaluate()
