import torch
import pandas as pd
from src.torch_indicators import EMA, SMA, STDP, REF, HHV, LLV

def extract_features(df: pd.DataFrame):
    """
    Extracts features from the SPY dataframe using PyTorch for indicator calculations.
    Matches logic from x-3.txt.
    """
    # Convert required columns to PyTorch Tensors
    C = torch.tensor(df['Close'].values, dtype=torch.float32)
    H = torch.tensor(df['High'].values, dtype=torch.float32)
    L = torch.tensor(df['Low'].values, dtype=torch.float32)
    
    # Extract dates to attach later if needed
    dates = df.index if 'Date' not in df.columns else df['Date']
    
    # ----------------------------------------------------
    # TEMA Short
    # ----------------------------------------------------
    P3 = 18
    EMA1 = EMA(C, P3)
    EMAA2 = EMA(EMA1, P3)
    EMA3 = EMA(EMAA2, P3)
    DIS = STDP(C, P3)
    
    TEU3 = (3 * (EMA1 - EMAA2) + EMA3) + DIS
    TEMA3 = 3 * (EMA1 - EMAA2) + EMA3
    TED = (3 * (EMA1 - EMAA2) + EMA3) - DIS
    
    NP1 = 6
    TEMA3T3 = (TEMA3 - REF(TEMA3, NP1)) / REF(TEMA3, NP1)
    TEMA3T1 = (TEMA3 - REF(TEMA3, 1)) / REF(TEMA3, 1)
    
    # ----------------------------------------------------
    # TEMA Long
    # ----------------------------------------------------
    P2 = 80
    EMAP21 = EMA(C, P2)
    EMAP221 = EMA(EMAP21, P2)
    EMAP231 = EMA(EMAP221, P2)
    TEMAP2 = 3 * EMAP21 - 3 * EMAP221 + EMAP231
    TEMA3T2 = (TEMAP2 - REF(TEMAP2, NP1)) / REF(TEMAP2, NP1)
    
    # ----------------------------------------------------
    # KDJ Long
    # ----------------------------------------------------
    NJ1, MJ1, PJ1 = 204, 80, 12
    RSV1 = (C - LLV(L, NJ1)) / (HHV(H, NJ1) - LLV(L, NJ1)) * 100
    K1 = SMA(RSV1, MJ1, 1)
    D1 = SMA(K1, PJ1, 1)
    J1 = 3 * K1 - 2 * D1
    
    # ----------------------------------------------------
    # KDJ Short
    # ----------------------------------------------------
    NJ2, MJ2, PJ2 = 18, 7, 6
    RSV2 = (C - LLV(L, NJ2)) / (HHV(H, NJ2) - LLV(L, NJ2)) * 100
    K2 = SMA(RSV2, MJ2, 1)
    D2 = SMA(K2, PJ2, 1)
    J2 = 3 * K2 - 2 * D2
    
    # ----------------------------------------------------
    # KDJ Micro
    # ----------------------------------------------------
    NJ3, MJ3, PJ3 = 9, 5, 3
    RSV3 = (C - LLV(L, NJ3)) / (HHV(H, NJ3) - LLV(L, NJ3)) * 100
    K3 = SMA(RSV3, MJ3, 1)
    D3 = SMA(K3, PJ3, 1)
    J3 = 3 * K3 - 2 * D3
    
    # ----------------------------------------------------
    # KDJ N3=36 (For SP2 Exhaustion Signal)
    # ----------------------------------------------------
    N3, M3, M31 = 36, 18, 3
    RSV3_36 = (C - LLV(L, N3)) / (HHV(H, N3) - LLV(L, N3)) * 100
    KN3_36 = SMA(RSV3_36, M3, 1)
    DN3_36 = SMA(KN3_36, M31, 1)
    JN3_36 = 3 * KN3_36 - 2 * DN3_36
    
    # ----------------------------------------------------
    # Composite Indicator JX (Refactored for Linear Factorization)
    # The original formula: JX = J1 + J2 - 50 + J2 * TEMA3T3 * 6 + J1 * TEMA3T2 * 6 + J3 * TEMA3T1
    # We strip the -50 and 6 constants so the network can learn them:
    # JX_base = J1 + J2 + J3 * TEMA3T1
    # F1 = J2 * TEMA3T3
    # F2 = J1 * TEMA3T2
    # ----------------------------------------------------
    JX_base = J1 + J2 + J3 * TEMA3T1
    F1 = J2 * TEMA3T3
    F2 = J1 * TEMA3T2
    
    # Precompute the EMA pieces because EMA is a linear operator
    # EMA(JX_dynamic) = EMA(JX_base) + W_bias + W1*EMA(F1) + W2*EMA(F2)
    EMA_JX_base = EMA(JX_base, 5)
    EMA_F1 = EMA(F1, 5)
    EMA_F2 = EMA(F2, 5)
    
    # We also need EMA8 versions for other specific strategy references if they exist
    # If they are not used immediately, we precompute them to maintain standard backward comp.
    EMA8_JX_base = EMA(JX_base, 8)
    EMA8_F1 = EMA(F1, 8)
    EMA8_F2 = EMA(F2, 8)
    
    # Temporarily calculate a "Default JX" to satisfy the existing BARSLAST logic 
    # for `ma_c_down` and `ma_c_up` until we can move that logic into DifferentiableExpertModel.
    # Note: These constraints will be fixed at the default -50 and 6 for this batch, but JX evaluation
    # inside DifferentiableExpertModel will use the dynamic learned weights.
    JX_default = JX_base - 50 + F1 * 6 + F2 * 6
    EMAJX_default = EMA_JX_base - 50 + EMA_F1 * 6 + EMA_F2 * 6
    EMAJX8_default = EMA8_JX_base - 50 + EMA8_F1 * 6 + EMA8_F2 * 6
    
    # ----------------------------------------------------
    # MA_DYNAMIC (Cross dependencies for BK2 / SK2)
    # ----------------------------------------------------
    from src.torch_indicators import CROSSDOWN, CROSS, BARSLAST, MA_DYNAMIC
    barslast_crossdown = BARSLAST(CROSSDOWN(JX_default, EMAJX_default))
    ma_c_down = MA_DYNAMIC(C, barslast_crossdown)
    
    barslast_cross = BARSLAST(CROSS(JX_default, EMAJX_default))
    ma_c_up = MA_DYNAMIC(C, barslast_cross)

    # Gather everything into a dictionary
    features = {
        'C': C, 'H': H, 'L': L,
        'TEU3': TEU3, 'TEMA3': TEMA3, 'TED': TED,
        'TEMA3T1': TEMA3T1, 'TEMA3T2': TEMA3T2, 'TEMA3T3': TEMA3T3,
        'K1': K1, 'D1': D1, 'J1': J1,
        'K2': K2, 'D2': D2, 'J2': J2,
        'K3': K3, 'D3': D3, 'J3': J3,
        'JN3_36': JN3_36,
        'JX_base': JX_base, 'F1': F1, 'F2': F2,
        'EMA_JX_base': EMA_JX_base, 'EMA_F1': EMA_F1, 'EMA_F2': EMA_F2,
        'ma_c_down': ma_c_down, 'ma_c_up': ma_c_up,
        # Default fallbacks
        'JX': JX_default, 'EMAJX': EMAJX_default, 'EMAJX8': EMAJX8_default
    }
    
    return features
