"""
Differentiable Feature Extraction Pipeline.

Replaces the static features_torch.py computation with a nn.Module that has
LEARNABLE EMA/SMA smoothing alphas.  All indicator computations (KDJ, TEMA,
composite JX) flow through these learnable parameters, enabling end-to-end
gradient propagation from the loss back into the indicator internals.

Learnable parameters (20 total):
  - 6 EMA alphas for TEMA (3 short P3=18 + 3 long P2=80)
  - 8 SMA alphas for KDJ (K/D pair × 4 KDJ variants: Long/Short/Micro/N3)
  - 6 EMA alphas for JX EMA components (period-5 and period-8, ×3 each)

Non-learnable (fixed) operations:
  - HHV/LLV with fixed window sizes (204, 18, 9, 36)
  - STDP (population std deviation)
  - CROSS/CROSSDOWN/BARSLAST/MA_DYNAMIC for ma_c_down/ma_c_up

At initialization every alpha matches the original MyLanguage strategy exactly,
so the feature extractor reproduces the same indicators at Epoch 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.torch_indicators import HHV, LLV, STDP, CROSS, CROSSDOWN, BARSLAST, MA_DYNAMIC


class DiffEMA(nn.Module):
    """
    Differentiable EMA/SMA via causal conv1d with learnable alpha.

    Stores alpha in logit space: alpha = sigmoid(w_alpha).
    The kernel is an exponential decay:  w_k = alpha * (1-alpha)^k
    reconstructed on each forward pass so gradients flow to w_alpha.

    Works for both EMA (init_alpha = 2/(N+1)) and SMA (init_alpha = M/N).
    """
    def __init__(self, init_alpha, max_kernel=None):
        super().__init__()
        init_alpha = max(min(init_alpha, 1.0 - 1e-6), 1e-6)
        self.w_alpha = nn.Parameter(torch.tensor(
            math.log(init_alpha / (1.0 - init_alpha)), dtype=torch.float32
        ))
        # Kernel long enough to capture 99.9% of total weight
        if max_kernel is None:
            # (1-alpha)^K < 0.001 → K > log(0.001)/log(1-alpha)
            K = int(-math.log(0.001) / (-math.log(1.0 - init_alpha) + 1e-12))
            max_kernel = max(min(K + 10, 800), 30)
        self.max_kernel = max_kernel

    def forward(self, x):
        """
        x: [T] 1-D tensor.
        Returns: [T] tensor (EMA/SMA of x with learnable alpha).
        """
        alpha = torch.sigmoid(self.w_alpha)

        k = torch.arange(self.max_kernel, device=x.device, dtype=x.dtype)
        kernel = alpha * (1.0 - alpha).pow(k)          # [K]
        w = kernel.flip(0).reshape(1, 1, -1)            # [1, 1, K] for conv1d

        # Pad LEFT with x[0] (matches standard EMA init y[0] = x[0]).
        # Zero-padding would give y[0] ≈ 0, causing multi-hundred-bar warm-up drift.
        x3 = x.unsqueeze(0).unsqueeze(0)                # [1, 1, T]
        left_pad = x[:1].expand(self.max_kernel - 1).unsqueeze(0).unsqueeze(0)
        x_padded = torch.cat([left_pad, x3], dim=2)     # [1, 1, K-1+T]
        out = F.conv1d(x_padded, w)                      # [1, 1, T] (no extra padding needed)
        return out.squeeze(0).squeeze(0)                  # [T]


def _ref_safe(x, n):
    """Shift x right by n bars; pad with x[0] instead of NaN."""
    if n <= 0:
        return x
    return torch.cat([x[:1].expand(n), x[:-n]])


class DifferentiableFeatureExtractor(nn.Module):
    """
    Computes all 30 canonical features from raw Close/High/Low tensors,
    with learnable EMA/SMA alphas for every smoothing operation.
    """
    def __init__(self):
        super().__init__()

        # ---- TEMA Short (P3 = 18) ----
        a18 = 2.0 / (18 + 1)
        self.ema_ts1 = DiffEMA(a18)
        self.ema_ts2 = DiffEMA(a18)
        self.ema_ts3 = DiffEMA(a18)

        # ---- TEMA Long (P2 = 80) ----
        a80 = 2.0 / (80 + 1)
        self.ema_tl1 = DiffEMA(a80)
        self.ema_tl2 = DiffEMA(a80)
        self.ema_tl3 = DiffEMA(a80)

        # ---- KDJ Long (N=204, M=80/1, P=12/1) ----
        self.sma_k1 = DiffEMA(1.0 / 80)   # SMA(RSV, 80, 1)
        self.sma_d1 = DiffEMA(1.0 / 12)   # SMA(K1, 12, 1)

        # ---- KDJ Short (N=18, M=7/1, P=6/1) ----
        self.sma_k2 = DiffEMA(1.0 / 7)
        self.sma_d2 = DiffEMA(1.0 / 6)

        # ---- KDJ Micro (N=9, M=5/1, P=3/1) ----
        self.sma_k3 = DiffEMA(1.0 / 5)
        self.sma_d3 = DiffEMA(1.0 / 3)

        # ---- KDJ N3=36 (M=18/1, P=3/1) ----
        self.sma_kn3 = DiffEMA(1.0 / 18)
        self.sma_dn3 = DiffEMA(1.0 / 3)

        # ---- JX EMA-5 components ----
        a5 = 2.0 / (5 + 1)
        self.ema_jx5_base = DiffEMA(a5)
        self.ema_jx5_f1 = DiffEMA(a5)
        self.ema_jx5_f2 = DiffEMA(a5)

        # ---- JX EMA-8 components ----
        a8 = 2.0 / (8 + 1)
        self.ema_jx8_base = DiffEMA(a8)
        self.ema_jx8_f1 = DiffEMA(a8)
        self.ema_jx8_f2 = DiffEMA(a8)

    def forward(self, C, H, L):
        """
        Args:
            C, H, L: [T] raw price tensors (already float32, no NaN).
        Returns:
            dict mapping feature names to [T] tensors (same keys as FEATURE_KEYS).
        """
        # ---- TEMA Short ----
        EMA1 = self.ema_ts1(C)
        EMA2 = self.ema_ts2(EMA1)
        EMA3 = self.ema_ts3(EMA2)
        TEMA3 = 3.0 * (EMA1 - EMA2) + EMA3

        DIS = STDP(C, 18)
        TEU3 = TEMA3 + DIS
        TED = TEMA3 - DIS

        NP1 = 6
        ref_tema3_6 = _ref_safe(TEMA3, NP1)
        ref_tema3_1 = _ref_safe(TEMA3, 1)
        TEMA3T3 = (TEMA3 - ref_tema3_6) / (ref_tema3_6.abs() + 1e-8)
        TEMA3T1 = (TEMA3 - ref_tema3_1) / (ref_tema3_1.abs() + 1e-8)

        # ---- TEMA Long ----
        EMAP21 = self.ema_tl1(C)
        EMAP221 = self.ema_tl2(EMAP21)
        EMAP231 = self.ema_tl3(EMAP221)
        TEMAP2 = 3.0 * EMAP21 - 3.0 * EMAP221 + EMAP231
        ref_temap2_6 = _ref_safe(TEMAP2, NP1)
        TEMA3T2 = (TEMAP2 - ref_temap2_6) / (ref_temap2_6.abs() + 1e-8)

        # ---- KDJ Long (window 204) ----
        hhv1 = HHV(H, 204)
        llv1 = LLV(L, 204)
        denom1 = (hhv1 - llv1).clamp(min=1e-8)
        RSV1 = ((C - llv1) / denom1 * 100.0).clamp(0, 100)
        K1 = self.sma_k1(RSV1)
        D1 = self.sma_d1(K1)
        J1 = 3.0 * K1 - 2.0 * D1

        # ---- KDJ Short (window 18) ----
        hhv2 = HHV(H, 18)
        llv2 = LLV(L, 18)
        denom2 = (hhv2 - llv2).clamp(min=1e-8)
        RSV2 = ((C - llv2) / denom2 * 100.0).clamp(0, 100)
        K2 = self.sma_k2(RSV2)
        D2 = self.sma_d2(K2)
        J2 = 3.0 * K2 - 2.0 * D2

        # ---- KDJ Micro (window 9) ----
        hhv3 = HHV(H, 9)
        llv3 = LLV(L, 9)
        denom3 = (hhv3 - llv3).clamp(min=1e-8)
        RSV3 = ((C - llv3) / denom3 * 100.0).clamp(0, 100)
        K3 = self.sma_k3(RSV3)
        D3 = self.sma_d3(K3)
        J3 = 3.0 * K3 - 2.0 * D3

        # ---- KDJ N3=36 ----
        hhv_n3 = HHV(H, 36)
        llv_n3 = LLV(L, 36)
        denom_n3 = (hhv_n3 - llv_n3).clamp(min=1e-8)
        RSV3_36 = ((C - llv_n3) / denom_n3 * 100.0).clamp(0, 100)
        KN3_36 = self.sma_kn3(RSV3_36)
        DN3_36 = self.sma_dn3(KN3_36)
        JN3_36 = 3.0 * KN3_36 - 2.0 * DN3_36

        # ---- Composite JX (factored form) ----
        TEMA3T1_safe = torch.nan_to_num(TEMA3T1, nan=0.0)
        TEMA3T2_safe = torch.nan_to_num(TEMA3T2, nan=0.0)
        TEMA3T3_safe = torch.nan_to_num(TEMA3T3, nan=0.0)

        JX_base = J1 + J2 + J3 * TEMA3T1_safe
        F1 = J2 * TEMA3T3_safe
        F2 = J1 * TEMA3T2_safe

        EMA_JX_base = self.ema_jx5_base(JX_base)
        EMA_F1 = self.ema_jx5_f1(F1)
        EMA_F2 = self.ema_jx5_f2(F2)

        EMA8_JX_base = self.ema_jx8_base(JX_base)
        EMA8_F1 = self.ema_jx8_f1(F1)
        EMA8_F2 = self.ema_jx8_f2(F2)

        # Default JX for ma_c_down/ma_c_up (fixed constants, non-differentiable path)
        JX_default = JX_base - 50.0 + F1 * 6.0 + F2 * 6.0
        EMAJX_default = EMA_JX_base - 50.0 + EMA_F1 * 6.0 + EMA_F2 * 6.0
        EMAJX8_default = EMA8_JX_base - 50.0 + EMA8_F1 * 6.0 + EMA8_F2 * 6.0

        # Detach for CROSS/BARSLAST (non-differentiable boolean ops)
        JX_det = JX_default.detach()
        EMAJX_det = EMAJX_default.detach()

        barslast_cd = BARSLAST(CROSSDOWN(JX_det, EMAJX_det))
        ma_c_down = MA_DYNAMIC(C.detach(), barslast_cd)

        barslast_cr = BARSLAST(CROSS(JX_det, EMAJX_det))
        ma_c_up = MA_DYNAMIC(C.detach(), barslast_cr)

        # Replace any residual NaN
        ma_c_down = torch.nan_to_num(ma_c_down, nan=0.0)
        ma_c_up = torch.nan_to_num(ma_c_up, nan=0.0)

        return {
            'C': C, 'H': H, 'L': L,
            'TEU3': TEU3, 'TEMA3': TEMA3, 'TED': TED,
            'TEMA3T1': TEMA3T1_safe, 'TEMA3T2': TEMA3T2_safe, 'TEMA3T3': TEMA3T3_safe,
            'K1': K1, 'D1': D1, 'J1': J1,
            'K2': K2, 'D2': D2, 'J2': J2,
            'K3': K3, 'D3': D3, 'J3': J3,
            'JN3_36': JN3_36,
            'JX_base': JX_base, 'F1': F1, 'F2': F2,
            'EMA_JX_base': EMA_JX_base, 'EMA_F1': EMA_F1, 'EMA_F2': EMA_F2,
            'ma_c_down': ma_c_down, 'ma_c_up': ma_c_up,
            'JX': JX_default, 'EMAJX': EMAJX_default, 'EMAJX8': EMAJX8_default,
        }
