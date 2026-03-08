import torch
import pandas as pd
from src.torch_indicators import SMA, REF, HHV, LLV, CROSS, CROSSDOWN, BARSLAST, EVERY, MA_DYNAMIC

def extract_labels(df: pd.DataFrame, features: dict, ast_config=None):
    """
    Extracts label markers using PyTorch based on the 'x-3 参考BK SP.txt' strategy rules.
    If ast_config is provided, JX and EMAJX are dynamically custom-calculated to exactly
    match the coefficients of the user's MyLanguage script, ensuring the Ground Truth
    labels reflect the provided strategy and not the system defaults.
    """
    C = features['C']
    H = features['H']
    L = features['L']
    
    J1 = features['J1']
    
    if ast_config:
        w_bias = ast_config.get('w_bias', -50.0)
        w_f1 = ast_config.get('w_f1', 6.0)
        w_f2 = ast_config.get('w_f2', 6.0)
        
        JX = features['JX_base'] + w_bias + features['F1'] * w_f1 + features['F2'] * w_f2
        EMAJX = features['EMA_JX_base'] + w_bias + features['EMA_F1'] * w_f1 + features['EMA_F2'] * w_f2
        # Use custom J1 weight for Cond3 if provided
        w_cond3_j1 = ast_config.get('w_cond3_j1', 1.0)
    else:
        JX = features['JX']
        EMAJX = features['EMAJX']
        w_cond3_j1 = 1.0
    # BK2 Point
    # ----------------------------------------------------
    cond1 = JX > REF(JX, 1)
    cond3 = JX > J1 * w_cond3_j1
    barslast_crossdown = BARSLAST(CROSSDOWN(JX, EMAJX))
    ma_c_down = MA_DYNAMIC(C, barslast_crossdown)
    cond2 = C > ma_c_down
    BK2 = (cond1 & cond2 & cond3).float()
    
    # ----------------------------------------------------
    # SK2 Point
    # ----------------------------------------------------
    cond4 = JX < REF(JX, 1)
    cond6 = JX < J1
    barslast_cross = BARSLAST(CROSS(JX, EMAJX))
    ma_c_up = MA_DYNAMIC(C, barslast_cross)
    cond5 = C < ma_c_up
    SK2 = (cond4 & cond5 & cond6).float()
    
    # ----------------------------------------------------
    # SP1 & BP1 Points
    # ----------------------------------------------------
    # BP1:=SK2=0&&REF(EVERY(SK2,3),1);
    # SP1:=BK2=0&&REF(EVERY(BK2,3),1);
    SP1 = ((BK2 == 0) & (REF(EVERY(BK2, 3), 1) == 1)).float()
    BP1 = ((SK2 == 0) & (REF(EVERY(SK2, 3), 1) == 1)).float()
    
    # ----------------------------------------------------
    # JN3 calculations for SP2 Point
    # ----------------------------------------------------
    N3, M3, M31 = 36, 18, 3
    RSV3 = (C - LLV(L, N3)) / (HHV(H, N3) - LLV(L, N3)) * 100
    KN3 = SMA(RSV3, M3, 1)
    DN3 = SMA(KN3, M31, 1)
    JN3 = 3 * KN3 - 2 * DN3
    
    # SP2:= JN3 < REF(JN3, 1)
    SP2 = (JN3 < REF(JN3, 1)).float()
    
    labels = {
        'BK2': BK2, 'SK2': SK2,
        'SP1': SP1, 'BP1': BP1,
        'SP2': SP2,
        'JN3': JN3
    }
    
    return labels
