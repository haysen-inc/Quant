import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys

app = Flask(__name__, static_folder='frontend')
CORS(app)
os.makedirs("models", exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/params', methods=['GET'])
def api_params():
    """Return parameter comparison: trained model vs original init."""
    try:
        import torch
        from src.differentiable_expert import DifferentiableExpertTransformer
        from collections import defaultdict

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DifferentiableExpertTransformer().to(device)
        sd = torch.load('models/quant_transformer.pth', map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        init_model = DifferentiableExpertTransformer().to(device)

        def g(m, p):
            o = m
            for s in p.split('.'):
                o = getattr(o, s)
            return round(o.item(), 6)

        def alph(m, n):
            return round(torch.sigmoid(dict(m.feature_extractor.named_parameters())[n + '.w_alpha']).item(), 6)

        def prd(a):
            return round(2.0 / a - 1, 1) if a > 0.001 else 9999

        # Layer 1: Decision constants
        constants = []
        for attr, name, desc in [
            ('w_bias', 'JX偏置', 'JX = J1+J2+[bias]+...'),
            ('w_f1', 'TEMA3T3权重', 'J2*TEMA3T3*[w_f1]'),
            ('w_f2', 'TEMA3T2权重', 'J1*TEMA3T2*[w_f2]'),
            ('w_cond3_j1', 'BK2入场门槛', 'JX > J1*[此值]'),
        ]:
            iv = g(init_model, f'expert_prior.{attr}')
            tv = g(model, f'expert_prior.{attr}')
            d = round(tv - iv, 6)
            pct = round(d / abs(iv) * 100, 2) if abs(iv) > 1e-8 else 0
            constants.append({'name': name, 'desc': desc, 'init': iv, 'trained': tv, 'delta': d, 'pct': pct})

        # Layer 1: Temperatures
        temps = []
        temp_meta = [
            ('dg_jx_rx',  'BK2: JX > REF(JX,1)',  'JX方向上升'),
            ('dg_jx_j1',  'BK2: JX > J1*w',       'JX高于J1'),
            ('dg_c_macd', 'BK2: C > MA_down',     '价格在均线上'),
            ('dl_jx_rx',  'SK2: JX < REF(JX,1)',   'JX方向下降'),
            ('dl_jx_j1',  'SK2: JX < J1',         'JX低于J1'),
            ('dl_c_macu', 'SK2: C < MA_up',       '价格在均线下'),
            ('dl_jn3',    'SP2: JN3 < REF(JN3,1)','KDJ_N3下降'),
        ]
        for gate, cond, desc in temp_meta:
            iv = g(init_model, f'expert_prior.{gate}.temperature')
            tv = g(model, f'expert_prior.{gate}.temperature')
            ratio = round(tv / iv, 1) if abs(iv) > 1e-8 else 0
            temps.append({'gate': gate, 'cond': cond, 'desc': desc, 'init': iv, 'trained': tv, 'ratio': ratio})

        # Layer 2: Factor alphas
        factor_groups = [
            ('TEMA短', '原周期18, 三层EMA级联', [('ema_ts1','EMA层1'),('ema_ts2','EMA层2'),('ema_ts3','EMA层3')]),
            ('TEMA长', '原周期80, 三层EMA级联', [('ema_tl1','EMA层1'),('ema_tl2','EMA层2'),('ema_tl3','EMA层3')]),
            ('KDJ长', 'N=204, K1/D1/J1', [('sma_k1','K线平滑'),('sma_d1','D线平滑')]),
            ('KDJ短', 'N=18, K2/D2/J2', [('sma_k2','K线平滑'),('sma_d2','D线平滑')]),
            ('KDJ微', 'N=9, K3/D3/J3', [('sma_k3','K线平滑'),('sma_d3','D线平滑')]),
            ('KDJ_N3', 'N=36, JN3止损', [('sma_kn3','K线平滑'),('sma_dn3','D线平滑')]),
            ('JX均线', 'EMA5/EMA8交叉', [
                ('ema_jx5_base','EMA5 base'),('ema_jx5_f1','EMA5 f1'),('ema_jx5_f2','EMA5 f2'),
                ('ema_jx8_base','EMA8 base'),('ema_jx8_f1','EMA8 f1'),('ema_jx8_f2','EMA8 f2'),
            ]),
        ]
        factors = []
        for title, desc, params in factor_groups:
            items = []
            for pn, pl in params:
                ia = alph(init_model, pn)
                ta = alph(model, pn)
                ip = prd(ia)
                tp = prd(ta)
                items.append({'name': pl, 'init_alpha': ia, 'trained_alpha': ta,
                              'init_period': ip, 'trained_period': tp, 'delta_period': round(tp - ip, 1)})
            factors.append({'title': title, 'desc': desc, 'params': items})

        # Layer 3: Transformer stats
        layer_stats = defaultdict(lambda: {'count': 0, 'max_delta': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0})
        init_dict = dict(init_model.named_parameters())
        for name, param in model.named_parameters():
            if name.startswith('expert_prior.') or name.startswith('feature_extractor.'):
                continue
            dd = (param - init_dict[name]).detach().cpu()
            parts = name.split('.')
            grp = '.'.join(parts[:3]) if len(parts) >= 3 else '.'.join(parts[:2])
            n = dd.numel()
            layer_stats[grp]['count'] += n
            layer_stats[grp]['max_delta'] = max(layer_stats[grp]['max_delta'], dd.abs().max().item())
            layer_stats[grp]['sum_sq'] += (dd ** 2).sum().item()
            layer_stats[grp]['sum_abs'] += dd.abs().sum().item()

        transformer = []
        for grp, s in sorted(layer_stats.items()):
            n = s['count']
            transformer.append({
                'name': grp, 'count': n,
                'mean_delta': round(s['sum_abs'] / n, 6),
                'max_delta': round(s['max_delta'], 6),
                'rms_delta': round((s['sum_sq'] / n) ** 0.5, 6),
            })

        total = sum(p.numel() for p in model.parameters())
        del model, init_model
        torch.cuda.empty_cache()

        return jsonify({
            'success': True,
            'decision_layer': {'constants': constants, 'temperatures': temps},
            'factor_layer': factors,
            'transformer_layer': transformer,
            'total_params': total,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/extract', methods=['GET'])
def api_extract():
    """Export trained parameters as MyLanguage strategy code."""
    try:
        from src.utils import extract_mylanguage_code
        code, err = extract_mylanguage_code('models/quant_transformer.pth')
        if err:
            return jsonify({'success': False, 'error': err}), 404
        return jsonify({'success': True, 'code': code})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ohlc', methods=['GET'])
def api_ohlc():
    """Return 2-year SPY OHLC + expert baseline + model baseline.
    Uses SAVED training data (not live yfinance) to ensure expert baseline is deterministic.
    """
    try:
        import datetime
        import os
        import numpy as np
        import torch
        import pandas as pd
        from src.features_torch import extract_features
        from src.labels_torch import extract_labels
        from src.dataset import LABEL_KEYS

        from src.data_loader import fetch_spy_data
        today = datetime.datetime.now()
        start_date = (today - datetime.timedelta(days=729)).strftime('%Y-%m-%d')
        df = fetch_spy_data(start_date=start_date, end_date=None, interval="1h")
        df.dropna(inplace=True)
        T = len(df)
        closes = df['Close'].values
        seq_len = 15
        warm_up = 210  # KDJ(204) needs ~210 bars; must match train.py

        # OHLC
        ohlc = []
        timestamps = []
        for i in range(T):
            ts = int(df.index[i].timestamp())
            timestamps.append(ts)
            ohlc.append({
                'time': ts,
                'open': round(float(df['Open'].iloc[i]), 2),
                'high': round(float(df['High'].iloc[i]), 2),
                'low': round(float(df['Low'].iloc[i]), 2),
                'close': round(float(df['Close'].iloc[i]), 2),
            })

        # Expert baseline labels
        fixed_features = extract_features(df)
        labels_dict = extract_labels(df, fixed_features)
        labels_np = np.stack([labels_dict[k].numpy() for k in LABEL_KEYS], axis=1)

        # ---- Model predictions (if checkpoint exists) ----
        model_np = None
        has_model = os.path.exists('models/quant_transformer.pth')
        if has_model:
            from src.differentiable_expert import DifferentiableExpertTransformer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DifferentiableExpertTransformer().to(device)
            sd = torch.load('models/quant_transformer.pth', map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            model.eval()
            C = torch.tensor(closes, dtype=torch.float32).to(device)
            H = torch.tensor(df['High'].values, dtype=torch.float32).to(device)
            L = torch.tensor(df['Low'].values, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs, _ = model(C, H, L, seq_len=seq_len)
            model_np = probs.cpu().numpy()
            del model
            torch.cuda.empty_cache()

        # ---- Run state machines from bar seq_len ----
        def _step_expert(state, yv, price, ts):
            bk2 = yv[0] == 1.0
            sk2 = yv[1] == 1.0
            sp1 = yv[3] == 1.0
            sp2 = yv[4] == 1.0
            bk2_edge = bk2 and not state['prev_bk2']
            if state['pos'] == 1:
                if sp1 or sp2 or sk2:
                    r = float(np.log(price / state['entry']))
                    state['trades'] += 1; state['pnl'] += r
                    if r > 0: state['wins'] += 1
                    state['trade_list'].append({
                        'entry_time': state['entry_time'], 'exit_time': ts,
                        'entry_price': round(float(state['entry']), 2),
                        'exit_price': round(float(price), 2),
                        'pnl_pct': round(r * 100, 3), 'win': bool(r > 0),
                    })
                    state['pos'] = 0
            if state['pos'] == 0 and bk2_edge:
                state['pos'] = 1; state['entry'] = price; state['entry_time'] = ts
            state['prev_bk2'] = bk2

        def _step_model(state, mp, price, ts):
            bk2 = mp[0] >= 0.5; sk2 = mp[1] >= 0.5
            sp1 = mp[3] >= 0.5; sp2 = mp[4] >= 0.5
            bk2_edge = bk2 and not state['prev_bk2']
            if state['pos'] == 1:
                if sp1 or sp2 or sk2:
                    r = float(np.log(price / state['entry']))
                    state['trades'] += 1; state['pnl'] += r
                    if r > 0: state['wins'] += 1
                    state['trade_list'].append({
                        'entry_time': state['entry_time'], 'exit_time': ts,
                        'entry_price': round(float(state['entry']), 2),
                        'exit_price': round(float(price), 2),
                        'pnl_pct': round(r * 100, 3), 'win': bool(r > 0),
                    })
                    state['pos'] = 0
            if state['pos'] == 0 and bk2_edge:
                state['pos'] = 1; state['entry'] = price; state['entry_time'] = ts
            state['prev_bk2'] = bk2

        expert = {'pos': 0, 'entry': 0, 'pnl': 0.0, 'wins': 0, 'trades': 0,
                  'prev_bk2': 0, 'entry_time': 0, 'trade_list': []}
        base_model = {'pos': 0, 'entry': 0, 'pnl': 0.0, 'wins': 0, 'trades': 0,
                      'prev_bk2': 0, 'entry_time': 0, 'trade_list': []}
        equity_expert = []
        equity_base = []

        for i in range(warm_up, T):
            price = closes[i]
            ts = timestamps[i]

            _step_expert(expert, labels_np[i], price, ts)
            equity_expert.append({'time': ts, 'value': round(float(expert['pnl'] * 100), 3)})

            if model_np is not None:
                _step_model(base_model, model_np[i], price, ts)
                equity_base.append({'time': ts, 'value': round(float(base_model['pnl'] * 100), 3)})

        def _stats(s):
            tr = s['trades']
            wr = round((s['wins'] / tr * 100), 1) if tr > 0 else 0.0
            return {'pnl': round(float(s['pnl'] * 100), 2), 'trades': tr, 'wr': wr,
                    'trade_list': s['trade_list']}

        result = {
            "success": True,
            "ohlc": ohlc,
            "expert": {**_stats(expert), "equity": equity_expert},
        }
        if has_model:
            result["base"] = {**_stats(base_model), "equity": equity_base}

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
