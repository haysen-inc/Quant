import pandas as pd
import datetime
from src.data_loader import fetch_spy_data
from src.features_torch import extract_features
from src.labels_torch import extract_labels

today = datetime.datetime.now()
start_date = (today - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
df = fetch_spy_data(start_date=start_date, end_date=None, interval="1h")
df.dropna(inplace=True)
features_dict = extract_features(df)
labels_dict = extract_labels(df, features_dict)

for k in ['BK2', 'SK2', 'BP1', 'SP1']:
    print(f"{k} fires {labels_dict[k].sum().item()} times out of {len(df)}")
