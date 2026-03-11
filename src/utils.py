import torch
import os
import math
from src.differentiable_expert import DifferentiableExpertTransformer


def extract_mylanguage_code(model_path):
    """
    Extract ALL learned parameters from checkpoint and generate
    complete, copy-pasteable MyLanguage (文华财经/TB) strategy code.

    Exports:
    - 20 EMA/SMA period parameters (learned from differentiable feature extractor)
    - 4 JX formula constants (w_bias, w_f1, w_f2, w_cond3_j1)
    - Comments showing original vs trained values
    """
    if not os.path.exists(model_path):
        return None, "Model file not found"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DifferentiableExpertTransformer().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # ---- Extract expert prior constants ----
    w_bias = model.expert_prior.w_bias.item()
    w_f1 = model.expert_prior.w_f1.item()
    w_f2 = model.expert_prior.w_f2.item()
    w_cond3_j1 = model.expert_prior.w_cond3_j1.item()

    # ---- Extract learned EMA/SMA periods ----
    def get_alpha(name):
        return torch.sigmoid(dict(model.feature_extractor.named_parameters())[name + '.w_alpha']).item()

    def ema_period(alpha):
        """EMA(x, N): alpha = 2/(N+1) → N = 2/alpha - 1"""
        return max(2, round(2.0 / alpha - 1))

    def sma_period(alpha):
        """SMA(x, M, 1): alpha = 1/M → M = 1/alpha"""
        return max(2, round(1.0 / alpha))

    # TEMA短: 3 EMA layers (original all P3=18)
    ts1 = ema_period(get_alpha('ema_ts1'))  # EMA of C
    ts2 = ema_period(get_alpha('ema_ts2'))  # EMA of EMA
    ts3 = ema_period(get_alpha('ema_ts3'))  # EMA of EMA^2

    # TEMA长: 3 EMA layers (original all P2=80)
    tl1 = ema_period(get_alpha('ema_tl1'))
    tl2 = ema_period(get_alpha('ema_tl2'))
    tl3 = ema_period(get_alpha('ema_tl3'))

    # KDJ长: SMA smoothing (original MJ1=80, PJ1=12)
    mj1 = sma_period(get_alpha('sma_k1'))
    pj1 = sma_period(get_alpha('sma_d1'))

    # KDJ短: SMA smoothing (original MJ2=7, PJ2=6)
    mj2 = sma_period(get_alpha('sma_k2'))
    pj2 = sma_period(get_alpha('sma_d2'))

    # KDJ微: SMA smoothing (original MJ3=5, PJ3=3)
    mj3 = sma_period(get_alpha('sma_k3'))
    pj3 = sma_period(get_alpha('sma_d3'))

    # KDJ_N3: SMA smoothing (original M3=18, M31=3)
    m3 = sma_period(get_alpha('sma_kn3'))
    m31 = sma_period(get_alpha('sma_dn3'))

    # JX均线: EMA(JX, 5) and EMA(JX, 8)
    # These have 3 sub-components each (base, f1, f2), take the base as representative
    jx_ema5 = ema_period(get_alpha('ema_jx5_base'))
    jx_ema8 = ema_period(get_alpha('ema_jx8_base'))

    # ---- Format bias string ----
    bias_str = f"{w_bias:.4f}" if w_bias < 0 else f"+{w_bias:.4f}"

    # ---- Same or different? ----
    tema_s_same = (ts1 == ts2 == ts3)
    tema_l_same = (tl1 == tl2 == tl3)

    # ---- Generate full MyLanguage code ----
    lines = []
    lines.append("// ========== AI 深度学习优化版策略 ==========")
    lines.append(f"// 模型: {os.path.basename(model_path)}")
    lines.append(f"// 专家常数: bias={w_bias:.4f} f1={w_f1:.4f} f2={w_f2:.4f} j1_gate={w_cond3_j1:.4f}")
    lines.append("")

    # TEMA短
    if tema_s_same:
        lines.append(f"P3:={ts1};  // TEMA短周期 (原18)")
        lines.append("EMA1:=EMA(C,P3);")
        lines.append("EMAA2:=EMA(EMA1,P3);")
        lines.append("EMA3:=EMA(EMAA2,P3);")
    else:
        lines.append(f"// TEMA短: 三层EMA学到不同周期 (原统一18)")
        lines.append(f"P3_1:={ts1};  // 层1 (原18)")
        lines.append(f"P3_2:={ts2};  // 层2 (原18)")
        lines.append(f"P3_3:={ts3};  // 层3 (原18)")
        lines.append("EMA1:=EMA(C,P3_1);")
        lines.append("EMAA2:=EMA(EMA1,P3_2);")
        lines.append("EMA3:=EMA(EMAA2,P3_3);")

    lines.append("DIS:=STDP(CLOSE,P3_1);") if not tema_s_same else lines.append("DIS:=STDP(CLOSE,P3);")
    lines.append("TEU3:=(3*(EMA1-EMAA2)+EMA3)+DIS;")
    lines.append("TEMA3:=3*(EMA1-EMAA2)+EMA3;")
    lines.append("TED:=(3*(EMA1-EMAA2)+EMA3)-DIS;")
    lines.append("")

    # TEMA升角
    lines.append("NP1:=6;")
    lines.append("TEMA3T3:=(TEMA3-REF(TEMA3,NP1))/REF(TEMA3,NP1);")
    lines.append("TEMA3T1:=(TEMA3-REF(TEMA3,1))/REF(TEMA3,1);")
    lines.append("")

    # TEMA长
    if tema_l_same:
        lines.append(f"P2:={tl1};  // TEMA长周期 (原80)")
        lines.append("EMAP21:=EMA(C,P2);")
        lines.append("EMAP221:=EMA(EMAP21,P2);")
        lines.append("EMAP231:=EMA(EMAP221,P2);")
    else:
        lines.append(f"// TEMA长: 三层EMA学到不同周期 (原统一80)")
        lines.append(f"P2_1:={tl1};  // 层1 (原80)")
        lines.append(f"P2_2:={tl2};  // 层2 (原80)")
        lines.append(f"P2_3:={tl3};  // 层3 (原80)")
        lines.append("EMAP21:=EMA(C,P2_1);")
        lines.append("EMAP221:=EMA(EMAP21,P2_2);")
        lines.append("EMAP231:=EMA(EMAP221,P2_3);")

    lines.append("TEMAP2:=3*EMAP21-3*EMAP221+EMAP231;")
    lines.append("TEMA3T2:=(TEMAP2-REF(TEMAP2,NP1))/REF(TEMAP2,NP1);")
    lines.append("")

    # KDJ长
    lines.append(f"NJ1:=204;")
    lines.append(f"MJ1:={mj1};  // K线平滑 (原80)")
    lines.append(f"PJ1:={pj1};  // D线平滑 (原12)")
    lines.append("RSV1:=(CLOSE-LLV(LOW,NJ1))/(HHV(HIGH,NJ1)-LLV(LOW,NJ1))*100;")
    lines.append("K1:SMA(RSV1,MJ1,1);")
    lines.append("D1:SMA(K1,PJ1,1);")
    lines.append("J1:3*K1-2*D1;")
    lines.append("")

    # KDJ短
    lines.append(f"NJ2:=18;")
    lines.append(f"MJ2:={mj2};  // K线平滑 (原7)")
    lines.append(f"PJ2:={pj2};  // D线平滑 (原6)")
    lines.append("RSV2:=(CLOSE-LLV(LOW,NJ2))/(HHV(HIGH,NJ2)-LLV(LOW,NJ2))*100;")
    lines.append("K2:=SMA(RSV2,MJ2,1);")
    lines.append("D2:=SMA(K2,PJ2,1);")
    lines.append("J2:=3*K2-2*D2;")
    lines.append("")

    # KDJ微
    lines.append(f"NJ3:=9;")
    lines.append(f"MJ3:={mj3};  // K线平滑 (原5)")
    lines.append(f"PJ3:={pj3};  // D线平滑 (原3)")
    lines.append("RSV3:=(CLOSE-LLV(LOW,NJ3))/(HHV(HIGH,NJ3)-LLV(LOW,NJ3))*100;")
    lines.append("K3:=SMA(RSV3,MJ3,1);")
    lines.append("D3:=SMA(K3,PJ3,1);")
    lines.append("J3:=3*K3-2*D3;")
    lines.append("")

    # JX (trained constants)
    lines.append(f"// JX公式: bias={w_bias:.4f} (原-50), f1={w_f1:.4f} (原6), f2={w_f2:.4f} (原6)")
    lines.append(f"JX:=J1+J2{bias_str}+J2*TEMA3T3*({w_f1:.4f})+J1*TEMA3T2*({w_f2:.4f})+J3*TEMA3T1,DOT;")
    lines.append("")

    # EMAJX
    lines.append(f"EMAJX:EMA(JX,{jx_ema5});  // (原5)")
    lines.append(f"EMAJX8:EMA(JX,{jx_ema8});  // (原8)")
    lines.append("")

    # BK2 / SK2 with learned j1 gate
    lines.append(f"// BK2入场门槛: JX > J1 * {w_cond3_j1:.4f} (原1.0)")
    lines.append(f"BK2:=JX>REF(JX,1)&&C>MA(C,BARSLAST(CROSSDOWN(JX,EMAJX)))&&JX>J1*{w_cond3_j1:.4f};")
    lines.append("DRAWICON(BK2,JX+0.2,'ICO6');")
    lines.append("SK2:=JX<REF(JX,1)&&C<MA(C,BARSLAST(CROSS(JX,EMAJX)))&&JX<J1;")
    lines.append("DRAWICON(SK2,JX+0.2,'ICO7');")
    lines.append("SP1:=BK2=0&&REF(EVERY(BK2,3),1);")
    lines.append("DRAWICON(SP1,JX+0.2,'ICO11');")
    lines.append("BP1:=SK2=0&&REF(EVERY(SK2,3),1);")
    lines.append("DRAWICON(BP1,JX-35,'ICO10');")
    lines.append("")

    # SP2 (KDJ_N3)
    lines.append(f"N3:=36;")
    lines.append(f"M3:={m3};  // K线平滑 (原18)")
    lines.append(f"M31:={m31};  // D线平滑 (原3)")
    lines.append("RSV3N:=(CLOSE-LLV(LOW,N3))/(HHV(HIGH,N3)-LLV(LOW,N3))*100;")
    lines.append("KN3:=SMA(RSV3N,M3,1);")
    lines.append("DN3:=SMA(KN3,M31,1);")
    lines.append("JN3:=3*KN3-2*DN3;")
    lines.append("SP2:=JN3<REF(JN3,1);")

    del model
    torch.cuda.empty_cache()

    return "\n".join(lines), None
