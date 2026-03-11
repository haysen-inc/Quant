import pandas as pd
import numpy as np
import torch
import os
from src.data_loader import fetch_spy_data
from src.features_torch import extract_features
from src.labels_torch import extract_labels
import datetime


def expert_state_machine(labels_array, closes_array, start_idx=0):
    """
    Run the MyLanguage expert state machine on sequential label/price arrays.

    Args:
        labels_array: [N, 5] numpy array of [BK2, SK2, BP1, SP1, SP2] labels
        closes_array: [N] numpy array of close prices (same length as labels_array)
        start_idx: skip first N bars (warm-up period, default 0 = no skip)

    Returns:
        dict with pnl, wins, trades, long_pnl, short_pnl, etc.

    Signal semantics: BK2=开多 SK2=开空 BP1=平空 SP1=平多 SP2=平多
    Execution rules:
      - Entry: BK2 edge → long, SK2 edge → short (edge detection prevents 重复开仓)
      - Exit: SP1/SP2 → close long, BP1 → close short
      - Reverse: SK2 active while long → close+reverse, BK2 active while short → close+reverse
      - No same-bar close+re-open in same direction
    """
    state = {'pos': 0, 'entry': 0.0, 'pnl': 0.0, 'wins': 0, 'trades': 0,
             'long_pnl': 0.0, 'short_pnl': 0.0, 'long_trades': 0, 'short_trades': 0,
             'long_wins': 0, 'short_wins': 0}
    prev_y = np.zeros(5)

    for i in range(start_idx, len(labels_array)):
        price = closes_array[i]
        bk2, sk2, bp1, sp1, sp2 = labels_array[i]
        bk2_edge = bk2 and not prev_y[0]
        sk2_edge = sk2 and not prev_y[1]
        closed = False

        # EXIT + REVERSE (priority over entry)
        if state['pos'] == 1:
            rev = 0
            should_exit = False
            if sp1 or sp2:
                should_exit = True
            elif sk2:
                should_exit = True; rev = -1
            if should_exit:
                r = np.log(price / state['entry'])
                state['trades'] += 1; state['long_trades'] += 1
                state['pnl'] += r; state['long_pnl'] += r
                if r > 0: state['wins'] += 1; state['long_wins'] += 1
                state['pos'] = 0; closed = True
                if rev: state['pos'] = rev; state['entry'] = price

        elif state['pos'] == -1:
            rev = 0
            should_exit = False
            if bp1:
                should_exit = True
            elif bk2:
                should_exit = True; rev = 1
            if should_exit:
                r = np.log(price / state['entry'])
                state['trades'] += 1; state['short_trades'] += 1
                state['pnl'] -= r; state['short_pnl'] -= r
                if r < 0: state['wins'] += 1; state['short_wins'] += 1
                state['pos'] = 0; closed = True
                if rev: state['pos'] = rev; state['entry'] = price

        # ENTRY (only from flat, edge detection, no same-bar re-open)
        if state['pos'] == 0 and not closed:
            if bk2_edge and not sk2_edge:
                state['pos'] = 1; state['entry'] = price
            elif sk2_edge and not bk2_edge:
                state['pos'] = -1; state['entry'] = price

        prev_y = np.array([bk2, sk2, bp1, sp1, sp2])

    return state


def long_only_state_machine(labels_array, closes_array, start_idx=0):
    """
    Long-only state machine: no shorting.

    Entry: BK2 edge → open long
    Exit:  SP1/SP2 → close long, SK2 → close long (exit, NOT short)
    """
    state = {'pos': 0, 'entry': 0.0, 'pnl': 0.0, 'wins': 0, 'trades': 0}
    prev_bk2 = 0

    for i in range(start_idx, len(labels_array)):
        price = closes_array[i]
        bk2, sk2, bp1, sp1, sp2 = labels_array[i]
        bk2_edge = bk2 and not prev_bk2

        if state['pos'] == 1:
            should_exit = False
            if sp1 or sp2:
                should_exit = True
            elif sk2:
                should_exit = True
            if should_exit:
                r = np.log(price / state['entry'])
                state['trades'] += 1
                state['pnl'] += r
                if r > 0: state['wins'] += 1
                state['pos'] = 0

        if state['pos'] == 0:
            if bk2_edge:
                state['pos'] = 1; state['entry'] = price

        prev_bk2 = bk2

    return state


def evaluate():
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=700)
    df = fetch_spy_data(start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), interval="1h")

    features_dict = extract_features(df)
    labels_dict = extract_labels(df, features_dict)

    y_bk2 = labels_dict['BK2'].numpy()
    y_sk2 = labels_dict['SK2'].numpy()
    y_bp1 = labels_dict['BP1'].numpy()
    y_sp1 = labels_dict['SP1'].numpy()
    y_sp2 = labels_dict['SP2'].numpy()

    labels_array = np.stack([y_bk2, y_sk2, y_bp1, y_sp1, y_sp2], axis=1)
    closes = df['Close'].values

    lo_state = long_only_state_machine(labels_array, closes, start_idx=15)

    wr = (lo_state['wins'] / lo_state['trades']) * 100 if lo_state['trades'] > 0 else 0
    print(f"--- Expert Baseline (Long-Only State Machine) ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Trades: {lo_state['trades']}  WR: {wr:.1f}%  PnL: {lo_state['pnl']*100:+.2f}%")

if __name__ == "__main__":
    evaluate()
