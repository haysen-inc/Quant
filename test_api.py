import requests
import json
print('Triggering API Pretrain request (1 Epoch) with AST...')
res = requests.post('http://127.0.0.1:5000/api/pretrain', json={
    'epochs': 1, 
    'learning_rate': 0.25, 
    'ast_config': {'w_bias': -50.0, 'w_cond3_j1': 1.0, 'w_f1': 6.0, 'w_f2': 6.0}
})
data = res.json()
m = data['metrics']
print(f'Model Win Rate: Model: {m["best_win_rate_percent"]:>4}% | Base: {m["best_base_win_rate_percent"]:>4}%')
print(f'PnL Outperformance: Model: {m["best_pnl_percent"]:>+5.2f}% | Base: {m["best_base_pnl_percent"]:>+5.2f}%')
print('\nEnd Evaluation Test.')
