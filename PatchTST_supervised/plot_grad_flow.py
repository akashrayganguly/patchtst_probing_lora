"""
plot_grad_flow.py
=================
Turn the diagnostic CSVs written by utils/grad_tracker.py into the two views
you asked for:

  1. ACROSS THE ARCHITECTURE for a given step  (a single CSV row)
     -> bar chart over modules for the chosen metric at one global_step.

  2. ACROSS ITERATIONS / EPOCHS for each module (each CSV column)
     -> line chart, one line per module, x = global_step.

Usage
-----
    # one panel per metric, both views, saved as PNGs:
    python plot_grad_flow.py --log_dir ./grad_logs/<setting>

    # just one metric, the trace view:
    python plot_grad_flow.py --log_dir ./grad_logs/<setting> \
        --metric grad_effective --view trace

Metrics available (one CSV each):
    grad_param_norm, grad_effective, grad_inflow, grad_outflow,
    fwd_branch_ratio, drift_rel_fro, drift_cosine, drift_norm_ratio,
    cap_stable_rank, cap_spectral_entropy, cap_effective_rank
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


META = ['epoch', 'global_step', 'iter', 'lr']

ALL_METRICS = [
    'grad_param_norm', 'grad_effective', 'grad_inflow', 'grad_outflow',
    'fwd_branch_ratio', 'drift_rel_fro', 'drift_cosine', 'drift_norm_ratio',
    'cap_stable_rank', 'cap_spectral_entropy', 'cap_effective_rank',
]


def _module_cols(df):
    return [c for c in df.columns if c not in META]


def plot_trace(df, metric, out_path, logy=False):
    cols = _module_cols(df)
    x = df['global_step'].values
    plt.figure(figsize=(11, 6))
    for c in cols:
        plt.plot(x, df[c].values, label=c, linewidth=1.6, alpha=0.9)
    plt.xlabel('global step')
    plt.ylabel(metric)
    if logy:
        plt.yscale('log')
    plt.title(f'{metric} across iterations (per module)')
    plt.legend(fontsize=7, ncol=2, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('wrote', out_path)


def plot_snapshot(df, metric, out_path, step=None):
    cols = _module_cols(df)
    if step is None:
        row = df.iloc[-1]            # last sampled step
    else:
        row = df.iloc[(df['global_step'] - step).abs().argmin()]
    vals = [row[c] for c in cols]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(cols)), vals)
    plt.xticks(range(len(cols)), cols, rotation=60, ha='right', fontsize=8)
    plt.ylabel(metric)
    plt.title(f'{metric} across architecture @ global_step={int(row["global_step"])}')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('wrote', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True, help='dir containing the metric CSVs')
    ap.add_argument('--metric', default='all',
                    help="metric name or 'all'")
    ap.add_argument('--view', default='both', choices=['trace', 'snapshot', 'both'])
    ap.add_argument('--step', type=int, default=None,
                    help='global_step for snapshot view (default: last)')
    ap.add_argument('--out_dir', default=None, help='where to save PNGs (default: log_dir/plots)')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    metrics = ALL_METRICS if args.metric == 'all' else [args.metric]
    # metrics that are easier to read on a log axis
    logy_set = {'grad_param_norm', 'grad_effective', 'grad_inflow', 'grad_outflow',
                'drift_rel_fro'}

    for metric in metrics:
        csv_path = os.path.join(args.log_dir, metric + '.csv')
        if not os.path.exists(csv_path):
            print('skip (missing):', csv_path)
            continue
        df = pd.read_csv(csv_path)
        if args.view in ('trace', 'both'):
            plot_trace(df, metric, os.path.join(out_dir, f'{metric}__trace.png'),
                       logy=(metric in logy_set))
        if args.view in ('snapshot', 'both'):
            plot_snapshot(df, metric, os.path.join(out_dir, f'{metric}__snapshot.png'),
                          step=args.step)


if __name__ == '__main__':
    main()
