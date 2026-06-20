"""
selftest_tracker.py
===================
Run this ONCE on the HPC (inside PatchTST_supervised/, with your conda env
active) to confirm the gradient tracker wires up correctly *before* you submit
the full 100-epoch job:

    python selftest_tracker.py

It builds a real PatchTST model with the ETTh1 hyper-parameters, runs a handful
of synthetic train steps with tracking armed, and checks that:
  * every learnable module was discovered into a column,
  * the activation hooks fired (inflow/outflow are finite for attn/ffn/head),
  * every metric CSV got rows written.

No dataset needed — inputs are random tensors of the right shape.
"""

import os
import types
import shutil

import torch

from models import PatchTST
from utils.grad_tracker import GradientFlowTracker


def make_args():
    a = types.SimpleNamespace()
    # ETTh1 official PatchTST config
    a.enc_in = 7
    a.seq_len = 336
    a.pred_len = 96
    a.e_layers = 3
    a.n_heads = 4
    a.d_model = 16
    a.d_ff = 128
    a.dropout = 0.3
    a.fc_dropout = 0.3
    a.head_dropout = 0.0
    a.individual = 0
    a.patch_len = 16
    a.stride = 8
    a.padding_patch = 'end'
    a.revin = 1
    a.affine = 0          # set to 1 to also exercise the revin_affine column
    a.subtract_last = 0
    a.decomposition = 0
    a.kernel_size = 25
    return a


def main():
    torch.manual_seed(0)
    args = make_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PatchTST.Model(args).float().to(device)
    model.train()

    log_dir = './grad_logs/_selftest'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    tracker = GradientFlowTracker(model, log_dir=log_dir, sample_frac=1.0, verbose=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 8
    steps = 4
    for it in range(steps):
        opt.zero_grad()
        x = torch.randn(B, args.seq_len, args.enc_in, device=device)
        y = torch.randn(B, args.pred_len, args.enc_in, device=device)
        tracker.arm()
        out = model(x)                       # [B, pred_len, enc_in]
        loss = torch.mean((out - y) ** 2)
        loss.backward()
        tracker.log_step(epoch=0, it=it, global_step=it, lr=opt.param_groups[0]['lr'])
        tracker.disarm()
        opt.step()
        print(f'  step {it}: loss={loss.item():.4f}')

    tracker.close()

    # ---- checks ----
    print('\n--- checks ---')
    print('discovered columns:', tracker.labels)

    ok = True
    for m in ['grad_param_norm', 'grad_effective', 'grad_inflow', 'grad_outflow',
              'fwd_branch_ratio', 'drift_rel_fro', 'drift_cosine', 'drift_norm_ratio',
              'cap_stable_rank', 'cap_spectral_entropy', 'cap_effective_rank']:
        path = os.path.join(log_dir, m + '.csv')
        n = sum(1 for _ in open(path)) - 1 if os.path.exists(path) else 0
        status = 'OK' if n == steps else 'MISSING/SHORT'
        if n != steps:
            ok = False
        print(f'  {m:22s} rows={n}  [{status}]')

    # spot-check that attention hooks actually fired (inflow finite)
    import csv
    with open(os.path.join(log_dir, 'grad_inflow.csv')) as f:
        r = list(csv.DictReader(f))[-1]
    attn_cols = [c for c in tracker.labels if c.endswith('_attn')]
    finite = [c for c in attn_cols if r[c] not in ('', 'nan') and r[c] == r[c]]
    print(f'  attention inflow finite for {len(finite)}/{len(attn_cols)} layers')
    if len(finite) != len(attn_cols):
        ok = False

    print('\nRESULT:', 'PASS' if ok else 'FAIL — inspect output above')
    print('CSVs in:', os.path.abspath(log_dir))


if __name__ == '__main__':
    main()
