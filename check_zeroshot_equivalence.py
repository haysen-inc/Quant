import torch
import pandas as pd
from src.features_torch import extract_features
from src.labels_torch import extract_labels
from src.differentiable_expert import DifferentiableExpertTransformer
from src.torch_indicators import BARSLAST, CROSSDOWN, CROSS, MA_DYNAMIC

def main():
    print("Loading SPY data for Zero-Shot Equivalence Test...")
    try:
        df = pd.read_csv('data/SPY_1d_historical.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run dataset script first.")
        return
        
    # Extract Base Features
    features = extract_features(df)
    C = features['C']
    JX = features['JX']
    EMAJX = features['EMAJX']
    
    # Extract the dynamic MAs (we'll precompute these for the network to use as raw inputs)
    barslast_crossdown = BARSLAST(CROSSDOWN(JX, EMAJX))
    ma_c_down = MA_DYNAMIC(C, barslast_crossdown)
    
    barslast_cross = BARSLAST(CROSS(JX, EMAJX))
    ma_c_up = MA_DYNAMIC(C, barslast_cross)
    
    # Attach these derived features to the input dict
    features['ma_c_down'] = ma_c_down
    features['ma_c_up'] = ma_c_up
    
    # 1. Run Ground Truth Expert Logic (Hard Boolean)
    labels = extract_labels(df, features)
    expert_bk2 = labels['BK2']
    expert_sk2 = labels['SK2']
    
    # 2. Add Batch & Sequence Dimensions to Features for the Model
    # Shape becomes [1, Seq_Len]
    model_inputs = {k: v.unsqueeze(0) for k, v in features.items() if isinstance(v, torch.Tensor)}
    
    # 3. Initialize Differentiable    
    # Initialize the model 
    model = DifferentiableExpertTransformer()
    model.eval() # Disable dropout and BN
    
    # We unpack the dict the same way train.py does
    # Run Forward Pass
    # Shape output [1, Seq_Len, 2] -> index 0 is BK2, 1 is SK2
    model_outs = model(model_inputs)
    model_bk2 = model_outs[0, :, 0]
    model_sk2 = model_outs[0, :, 1]
    
    # 4. Compare Equivalence
    # Since model uses Sigmoids (Soft logic), we threshold at 0.5 to check binary equivalence
    model_bk2_binary = (model_bk2 > 0.5).float()
    model_sk2_binary = (model_sk2 > 0.5).float()
    
    bk2_match_rate = (model_bk2_binary == expert_bk2).float().mean().item()
    sk2_match_rate = (model_sk2_binary == expert_sk2).float().mean().item()
    
    print(f"BK2 Zero-Shot Match Rate: {bk2_match_rate * 100:.2f}%")
    print(f"SK2 Zero-Shot Match Rate: {sk2_match_rate * 100:.2f}%")
    
    if bk2_match_rate == 1.0 and sk2_match_rate == 1.0:
        print("SUCCESS: Differentiable Model PERFECTLY mimics Hard Expert Rules at Epoch 0.")
    else:
        print("WARNING: Initialization failed to reach 100% equivalence.")

if __name__ == "__main__":
    main()
