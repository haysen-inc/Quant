import torch
import os
from src.differentiable_expert import DifferentiableExpertTransformer

import re

def extract_mylanguage_code(model_path, raw_code=None):
    """
    Extracts the deep learning parameters from a trained .pth checkpoint
    and formats them into copy-pasteable MyLanguage (文华/TB) syntax.
    If 'raw_code' is provided, it replaces the values in-place structurally.
    """
    if not os.path.exists(model_path):
        return None, "Model file not found"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DifferentiableExpertTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    w_bias = model.expert_prior.w_bias.item()
    w_f1 = model.expert_prior.w_f1.item()
    w_f2 = model.expert_prior.w_f2.item()
    w_cond3_j1 = model.expert_prior.w_cond3_j1.item()
    
    bias_str = f"{w_bias:.4f}" if w_bias < 0 else f"+{w_bias:.4f}"
    
    # Generate the Strict Replacement Block
    replacement_jx = f"JX:=J1+J2{bias_str}+J2*TEMA3T3*({w_f1:.4f})+J1*TEMA3T2*({w_f2:.4f})+J3*TEMA3T1;"
    
    if raw_code and isinstance(raw_code, str) and len(raw_code.strip()) > 10:
        # User provided the full script from the Frontend UI.
        # We will attempt to perform a direct inline replacement so they can just copy-paste the whole text window.
        
        # 1. Replace the entire JX:= line
        new_code = re.sub(r'JX:=?(.*?)[;,]', replacement_jx.replace(';', ''), raw_code, flags=re.IGNORECASE)
        
        # 2. Extract and replace the Cond3 weight inside BK2:=... AND JX > J1 * [W]
        # Regex looks for JX>J1 or JX>J1*W, taking accounting of potential spaces
        def bk2_repl(match):
            prefix = match.group(1)
            suffix = match.group(3)
            return f"{prefix}JX>J1*{w_cond3_j1:.4f}{suffix}"
            
        new_code = re.sub(r'(BK2.*?)(JX\s*>\s*J1(?:\s*\*\s*[\d\.\-]+)?)(.*?;)', bk2_repl, new_code, flags=re.IGNORECASE)
        
        # Append Debug block
        debug_info = f"\n// --- AI 深度学习进化参数 ({'RL Phase 14' if 'live_adapted' in model_path else 'Base Phase 12'}) ---\n"
        debug_info += f"// [调试] w_bias: {w_bias:.4f} | w_f1: {w_f1:.4f} | w_f2: {w_f2:.4f} | j1_factor: {w_cond3_j1:.4f}\n"
        
        return new_code + debug_info, None
    else:
        # Generate the classic syntax block if no raw code is supplied
        code = [
            "// AI 深度学习进化提纯版 (代码碎片)",
            "// ----------------------------------------------------",
            replacement_jx,
            f"BK2:= ... AND (JX > J1 * {w_cond3_j1:.4f});",
            "// ----------------------------------------------------",
            f"// [调试信息] w_bias: {w_bias:.4f} | w_f1: {w_f1:.4f} | w_f2: {w_f2:.4f} | j1_factor: {w_cond3_j1:.4f}"
        ]
        
        return "\n".join(code), None
