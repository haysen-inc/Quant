import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import fetch_spy_data
from features_torch import extract_features
from labels_torch import extract_labels

def main():
    print("Welcome to Quantitative Experiments with SPY!")
    print("Fetching initial data...")
    # Fetch SPY data from 2020 onwards for a quick test
    df = fetch_spy_data(start_date="2020-01-01", save_dir="data")
    
    print("\n--- Generating PyTorch Features ---")
    features = extract_features(df)
    print(f"Calculated {len(features)} feature tensors (length: {len(features['C'])}).")
    
    # Show a sample of TEMA3 and JX
    print(f"Sample TEMA3 (last 5): {features['TEMA3'][-5:]}")
    print(f"Sample JX (last 5): {features['JX'][-5:]}")
    
    print("\n--- Generating PyTorch Labels ---")
    labels = extract_labels(df, features)
    print(f"Calculated {len(labels)} label tensors.")
    
    # Check if we have any triggers
    for lbl in ['BK2', 'SK2', 'BP1', 'SP1', 'SP2']:
        trigger_count = (labels[lbl] == 1.0).sum().item()
        print(f"Total {lbl} triggers in dataset: {trigger_count}")

if __name__ == "__main__":
    main()
