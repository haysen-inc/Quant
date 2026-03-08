import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import sys

# Quant Pipeline Imports
from src.train import train_model
from src.online_rl_agent import run_live_simulation
from src.utils import extract_mylanguage_code
from src.mylanguage_parser import MyLanguageParser

app = Flask(__name__, static_folder='frontend')
CORS(app) # Enable cross-origin for frontend

# Ensure models directory exists for checkpoint saving
os.makedirs("models", exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/extract', methods=['POST', 'GET'])
def api_extract():
    # By default, load the base historically trained model
    data = request.json or {}
    model_type = request.args.get('type', 'base')
    raw_code = data.get('mylanguage', None)
    
    path = "models/live_adapted_expert_1h.pth" if model_type == 'rl' else "models/quant_transformer.pth"
    
    code, error = extract_mylanguage_code(path, raw_code=raw_code)
    if error:
        return jsonify({"success": False, "error": error}), 500
        
    return jsonify({"success": True, "mylanguage_code": code})

@app.route('/api/parse', methods=['POST'])
def api_parse():
    try:
        data = request.json or {}
        raw_code = data.get('mylanguage', '')
        
        # 1. Parse the JX formula base constants
        jx_result = MyLanguageParser.parse_jx_formula(raw_code)
        if not jx_result["success"]:
            return jsonify(jx_result), 400
            
        # 2. Parse the Cond3 (JX > J1 * w)
        bk_result = MyLanguageParser.parse_bk2_cond3(raw_code)
        w_cond3_j1 = bk_result.get("w_cond3_j1", 1.0) # Defaults to 1.0 if not found cleanly
        
        # Merge payload
        payload = jx_result["parsed_constants"]
        payload["w_cond3_j1"] = w_cond3_j1
        
        return jsonify({
            "success": True, 
            "message": "Strategy successfully parsed into Differentiable Tensors.",
            "ast_config": payload
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/pretrain', methods=['POST'])
def api_pretrain():
    try:
        data = request.json or {}
        epochs = int(data.get('epochs', 50))
        learning_rate = float(data.get('learning_rate', 0.25))
        ast_config = data.get('ast_config', None)
        raw_code = data.get('mylanguage', None)
        
        # Execute the heavy historical training process with AST Override
        print(f"--- API Triggered: Pre-training ({epochs} Epochs, {learning_rate} LR) ---")
        history = train_model(epochs=epochs, batch_size=128, interval="1h", ast_config=ast_config)
        
        # Extract the new MyLanguage parameters generated
        code, _ = extract_mylanguage_code("models/quant_transformer.pth", raw_code=raw_code)
        
        # Simplistic parsing of final metrics from training output log mechanism
        return jsonify({
            "success": True,
            "message": "Historical pre-training completed successfully.",
            "mylanguage_code": code,
            "metrics": {
                "final_val_loss": history['val_loss'][-1] if 'val_loss' in history else 0.0,
                "history_loss": history.get('train_loss', []),
                "history_w_bias": history.get('w_bias_history', []),
                "history_w_f1": history.get('w_f1_history', []),
                "history_w_f2": history.get('w_f2_history', []),
                "history_w_cond3_j1": history.get('w_cond3_j1_history', []),
                "history_temp_1": history.get('temp_dg_jx_rx', []),
                "history_temp_2": history.get('temp_dg_jx_j1', []),
                "history_temp_3": history.get('temp_dg_c_macd', []),
                "history_temp_4": history.get('temp_dl_jx_rx', []),
                "history_temp_5": history.get('temp_dl_c_macu', []),
                "best_pnl_percent": round(history.get('best_pnl', 0.0) * 100, 2),
                "best_win_rate_percent": round(history.get('best_win_rate', 0.0) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rl', methods=['POST'])
def api_rl():
    try:
        data = request.json or {}
        hold_period = int(data.get('hold_period', 35))
        
        raw_code = data.get('mylanguage', None)
        
        print(f"--- API Triggered: Online RL Live Adaptation (Hold Failsafe: {hold_period}) ---")
        final_rl_pnl, final_base_pnl, rl_trades, wr, rl_history, base_history, rl_decision_history = run_live_simulation()
        
        # Extract the latest parameters from the newly adapted model
        code, _ = extract_mylanguage_code("models/live_adapted_expert_1h.pth", raw_code=raw_code)
        
        return jsonify({
            "success": True,
            "message": "Online Reactive Learning stream adapted.",
            "mylanguage_code": code,
            "metrics": {
                "base_expert_pnl_percent": round(final_base_pnl * 100, 2),
                "rl_agent_pnl_percent": round(final_rl_pnl * 100, 2),
                "rl_trades_count": rl_trades,
                "rl_win_rate_percent": round(wr * 100, 2),
                "history_rl_pnl": rl_history,
                "history_base_pnl": base_history,
                "history_rl_decision": rl_decision_history
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
