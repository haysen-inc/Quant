import torch
import os
from src.differentiable_expert import DifferentiableExpertTransformer

def extract_and_translate(model_name, model_path):
    print(f"\n{'='*50}")
    print(f"解剖脑区权重: {model_name}")
    print(f"[{model_path}]")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print("未找到指定模型文件！")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DifferentiableExpertTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Extract structural weights
    w_bias = model.expert_prior.w_bias.item()
    w_f1 = model.expert_prior.w_f1.item()
    w_f2 = model.expert_prior.w_f2.item()
    w_cond3_j1 = model.expert_prior.w_cond3_j1.item()
    
    print("--- Extracted Learned Expert Parameters ---")
    print(f"w_bias: {w_bias:.4f} (Original: -50.0)")
    print(f"w_f1: {w_f1:.4f}   (Original: 6.0)")
    print(f"w_f2: {w_f2:.4f}   (Original: 6.0)")
    print(f"w_cond3_j1: {w_cond3_j1:.4f} (Original: 1.0)")
    
    print("\n--- [复制粘贴] 返回麦语言 (WenHua/TB) 代替人工拍脑袋公式 ---")
    print("// 原版手工盲猜配置:")
    print("// JX:=J1+J2-50+J2*TEMA3T3*6+J1*TEMA3T2*6+J3*TEMA3T1;")
    print("// BK2:= ... AND (JX > J1);")
    print("\n// AI 深度学习提纯版 (可直接替换):")
    # Format negative bias nicely
    bias_str = f"{w_bias:.4f}" if w_bias < 0 else f"+{w_bias:.4f}"
    print(f"JX:=J1+J2{bias_str}+J2*TEMA3T3*({w_f1:.4f})+J1*TEMA3T2*({w_f2:.4f})+J3*TEMA3T1;")
    print(f"BK2:= ... AND (JX > J1 * {w_cond3_j1:.4f});")

if __name__ == "__main__":
    extract_and_translate("预训练模型", "models/quant_transformer.pth")
