# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPY Differentiable Expert System — a quantitative trading system that converts traditional hardcoded trading indicators (MyLanguage/文华财经 syntax) into a fully differentiable deep learning pipeline. All boolean logic (`>`, `<`, crossovers) is replaced with soft sigmoid approximations so gradients flow end-to-end from loss through transformer, expert logic, and indicator computation.

## Commands

```bash
# Setup
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/dgxspark/Desktop/Quant
pip install -r requirements.txt

# Run web UI + API server (port 5000)
python app.py

# CLI feature/label extraction (standalone smoke test)
python main.py

# Verification scripts (no pytest suite)
python check_zeroshot_equivalence.py
python check_baseline_firing.py
python evaluate_baseline.py
python test_api.py

# TensorBoard
tensorboard --logdir=logs/diff_expert_1h
```

No linter, formatter, or type checker is configured.

## Architecture

### Unified Model (`DifferentiableExpertTransformer`)

One model, end-to-end: raw OHLC → learnable features → expert prior → transformer correction.

Forward dispatch:
- `model(C, H, L, seq_len)` → (all_probs[T,5], expert_probs[T,5]) — full-series end-to-end training
- `model(x_dict)` → [B,S,5] — inference from pre-computed features (backward compat)

### Learnable Parameter Hierarchy

1. **Feature Extractor** (20 params): EMA/SMA smoothing alphas for TEMA-18, TEMA-80, KDJ-Long/Short/Micro/N3, JX-EMA5/EMA8. Implemented as conv1d with exponentially decaying kernel reconstructed each forward pass. Initialized to match original MyLanguage constants exactly.

2. **Expert Prior** (11 params): JX formula constants (`w_bias=-50`, `w_f1=6`, `w_f2=6`, `w_cond3_j1=1`) + 7 sigmoid temperature scalars (one per soft-logic gate). `init_temp=0.5` balances gradient flow vs sharpness.

3. **Transformer** (~564K params): 2-layer encoder (d_model=64, nhead=4) with zero-initialized decoder. Provides residual correction in logit space: `final = sigmoid(logit(expert_probs) + correction)`.

### Key Design Decisions

- **Logit-space residual**: `sigmoid(logit(p) + correction)` instead of `clamp(p + correction, 0, 1)`. At init correction=0 → output = p (zero-shot equivalence). Gradient: `d/dp sigmoid(logit(p)) = 1` — never vanishes.

- **init_temp=0.5**: temp=10 or temp=100 causes sigmoid saturation (zero gradients for typical KDJ differences of 1-20). temp=0.5 gives usable gradients for |diff| < 10 while temperature learns to sharpen during training.

- **Conv1d EMA**: `DiffEMA` in `differentiable_features.py` implements EMA/SMA as causal convolution with kernel `α(1-α)^k`. Left-padded with x[0] (not zeros) to match standard EMA initialization. Gradients flow through the kernel to learnable `w_alpha`.

- **HHV/LLV windows**: Fixed (204, 18, 9, 36) — structural choices, not fine-tuned. CROSS/BARSLAST/MA_DYNAMIC detached from gradient graph (non-differentiable boolean ops).

- **Features NOT normalized** (no z-score) — intentional, expert logic operates on absolute price/indicator scales.

### Training (`train_model`)

Full-series forward pass per epoch. Three learning rate groups:
- Feature extractor alphas (`feature_lr=0.01`): conservative, avoid destabilizing indicators
- Expert constants + temps (`expert_lr=0.25`): moderate, tune strategy logic
- Transformer weights (`lr=0.01`): standard, learn residual corrections

BCE warm-up for first 3 epochs, then mixed loss (70% EV + 30% BCE). State machine checkpoint: converts model probs → hard signals → runs `long_only_state_machine()` → uses PnL (scaled by min trade count) as the checkpoint metric.

### Pipeline

1. **Historical Pre-Training** (`src/train.py`): `train_model()` — full-series end-to-end, all parameters.
2. **Web UI** (`app.py` + `frontend/`): Flask serves HTML/JS/CSS. TradingView Lightweight Charts for visualization.

### Loss Function

`AsymmetricExpertLoss` in `src/train.py`:
- **BCE mode** (epoch 0-2): Binary cross entropy with dynamic targets — losing expert signals get target forced to 0, profitable signals get amplified gradients (weight=50 for positive targets).
- **EV+BCE mode** (epoch 3+): Mixed 70% EV + 30% BCE. EV = `-λ * (probs * scaled_returns * signal_weights).mean()`. Conflict penalty (50.0), sparsity penalty (mean>0.3), anti-collapse penalty (mean<0.05).

### Data Flow

`src/data_loader.py` (yfinance) → `src/features_torch.py` (30-dim features, static, for labels) or `src/differentiable_features.py` (30-dim, learnable, inside model) → `src/labels_torch.py` (5 binary signals) → model.

### AST Injection

`src/mylanguage_parser.py` regex-extracts constants from MyLanguage source. Frontend POSTs to `/api/parse`, returns `ast_config` dict that initializes `nn.Parameter` slots.

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/parse` | POST | Parse MyLanguage code → AST config |
| `/api/pretrain` | POST | Historical training (epochs, lr, ast_config) |
| `/api/extract` | POST/GET | Extract trained params as MyLanguage code |
| `/` | GET | Serve frontend |

### Key Source Files

- `src/differentiable_features.py` — `DiffEMA`, `DifferentiableFeatureExtractor` (learnable indicator computation, 20 alpha params)
- `src/differentiable_expert.py` — `DiffGreater/DiffLess`, `DifferentiableExpertModel`, unified `DifferentiableExpertTransformer`
- `src/torch_indicators.py` — vectorized operators (EMA, SMA, HHV, LLV, CROSS, BARSLAST)
- `src/dataset.py` — `FEATURE_KEYS` (30 features), `LABEL_KEYS` (5 labels), `SPYSequenceDataset`
- `src/utils.py` — parameter extraction and MyLanguage code generation

## Common Issues

- **Zero gradients**: Check that `torch.clamp(x, 0, 1)` is NOT used on model outputs — use logit-space residual instead. Verify `init_temp` isn't too high (sigmoid saturation).
- **`ModuleNotFoundError: src`** — set `PYTHONPATH` to project root
- **NaN loss** — check indicator warmup periods; model uses `nan_to_num` but upstream NaN can propagate
- **500 from `/api/parse`** — MyLanguage regex failed; check formula syntax
- **Old checkpoints**: Use `strict=False` when loading — old checkpoints lack `feature_extractor.*` keys (defaults match original hardcoded params)
- Flask requires `flask-cors` (not in requirements.txt)
