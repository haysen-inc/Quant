import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch
import pandas as pd
from data_loader import fetch_spy_data
from features_torch import extract_features
from labels_torch import extract_labels
from dataset import SPYSequenceDataset

df = fetch_spy_data("2024-01-01", interval="1d")
features = extract_features(df)
labels = extract_labels(df, features)
ds = SPYSequenceDataset(features, labels, df)

print("Total elements in X_raw:", ds.X_raw.numel())
print("Total exact zeroes in X_raw:", (ds.X_raw == 0).sum().item())
feature_keys = ['TEU3', 'TEMA3', 'TED', 'TEMA3T1', 'TEMA3T2', 'TEMA3T3', 'K1', 'D1', 'J1', 'K2', 'D2', 'J2', 'K3', 'D3', 'J3', 'JX', 'EMAJX', 'EMAJX8']
for i, k in enumerate(feature_keys):
    zeros = (ds.X_raw[:, i] == 0).sum().item()
    if zeros > 0:
        print(f"Feature {k} has {zeros} ({zeros/len(ds.X_raw)*100:.1f}%) zeroes")
