"""
grad_tracker.py
================
Diagnostic instrumentation for PatchTST (and the Former models) to understand
*whether* and *how* the attention / FFN / linear-head / learnable-embedding
modules learn.

It logs, at a sampled cadence, three orthogonal families of quantities so you
can tell apart the three failure modes that all look like "ablating attention
doesn't change the loss":

  (A) Gradient REACHING a module        -> activation gradients via backward hooks
        - grad_inflow   = ||dL/d(output)||   (signal arriving from the loss side)
        - grad_outflow  = ||dL/d(input)||    (signal it passes further back)

  (B) Parameter MOVEMENT                -> parameter gradients + drift from init
        - grad_param_norm = ||dL/dW||  (summed in quadrature over the group)
        - grad_effective  = ||dL/dW|| / ||W||           (relative step proxy)
        - drift_rel_fro   = ||W - W0||_F / ||W0||_F      (how far weights moved)
        - drift_cosine    = cos(W, W0)
        - drift_norm_ratio= ||W|| / ||W0||

  (C) Functional CONTRIBUTION           -> forward branch ratio via forward hooks
        - fwd_branch_ratio = ||sublayer_out|| / ||residual_in||
          (for attn/ffn this directly explains an ablation that doesn't move loss)

  Capacity / expressivity (computed on singular values at sampled steps):
        - cap_stable_rank      = ||W||_F^2 / ||W||_2^2
        - cap_spectral_entropy = -sum p_i log p_i,  p_i = s_i^2 / sum s_j^2
        - cap_effective_rank   = exp(spectral_entropy)

Output: one CSV per metric in `log_dir`. In every file:
    columns = ['epoch','global_step','iter','lr', <module_1>, <module_2>, ...]
    rows    = one sampled training step.
So a single row is an across-architecture snapshot, and a single module column
is an across-iteration / across-epoch trace.

The tracker is deliberately self-contained and defensive: a failure inside any
metric is caught and written as NaN rather than killing training.
"""

import os
import re
import csv
import math

import torch


# ----------------------------------------------------------------------------- #
#  Module discovery
# ----------------------------------------------------------------------------- #
# Each "group" is a semantic module (e.g. "L0_attn") = a set of learnable weight
# tensors + (optionally) a container nn.Module we can hang activation hooks on.

_LAYER_ATTN = re.compile(r'^(?P<root>.+?\.encoder\.layers\.(?P<idx>\d+))\.self_attn\.')
_LAYER_FFN  = re.compile(r'^(?P<root>.+?\.encoder\.layers\.(?P<idx>\d+))\.ff\.')
_WP         = re.compile(r'^(?P<root>.+?)\.W_P\.(weight|bias)$')
_WPOS       = re.compile(r'^(?P<root>.+?)\.W_pos$')
_HEAD       = re.compile(r'^(?P<root>.+?\.head)\.(linear|linears)\b')
_REVIN      = re.compile(r'^(?P<root>.+?\.revin_layer)\.affine_(weight|bias)$')


def _prefix_label(name):
    """Decomposition produces model_res.* and model_trend.* ; tag them."""
    if name.startswith('model_res'):
        return 'res.'
    if name.startswith('model_trend'):
        return 'trend.'
    return ''


def classify(name):
    """
    Return (group_label, container_name_or_None, sort_key) for a parameter name,
    or None if we don't want to track this parameter.

    sort_key = (prefix_rank, kind_rank, layer_idx) for stable column ordering.
    """
    pfx = _prefix_label(name)
    prank = {'': 0, 'res.': 1, 'trend.': 2}[pfx]

    # sort_key = (prefix, bucket, layer_idx, kind) ; bucket 0=pre-layer, 1=layer, 2=head
    m = _LAYER_ATTN.match(name)
    if m:
        idx = int(m.group('idx'))
        return (f'{pfx}L{idx}_attn', m.group('root') + '.self_attn', (prank, 1, idx, 0))

    m = _LAYER_FFN.match(name)
    if m:
        idx = int(m.group('idx'))
        return (f'{pfx}L{idx}_ffn', m.group('root') + '.ff', (prank, 1, idx, 1))

    m = _WP.match(name)
    if m:
        return (f'{pfx}embed_patch', m.group('root') + '.W_P', (prank, 0, 0, 1))

    m = _WPOS.match(name)
    if m:
        return (f'{pfx}embed_pos', None, (prank, 0, 0, 2))

    m = _HEAD.match(name)
    if m:
        kind = m.group(2)
        container = (m.group('root') + '.linear') if kind == 'linear' else None
        return (f'{pfx}head_linear', container, (prank, 2, 0, 0))

    m = _REVIN.match(name)
    if m:
        return (f'{pfx}revin_affine', None, (prank, 0, 0, 0))

    return None


# ----------------------------------------------------------------------------- #
#  Per-group state
# ----------------------------------------------------------------------------- #
class _Group:
    def __init__(self, label, sort_key):
        self.label = label
        self.sort_key = sort_key
        self.params = []          # list of (pname, tensor)
        self.container_name = None
        # init snapshots (cpu, float32)
        self.init_snaps = {}      # pname -> tensor
        self.init_norm = 0.0
        # activation-side caches (refreshed each armed backward/forward)
        self.inflow = float('nan')
        self.outflow = float('nan')
        self.fwd_ratio = float('nan')

    def snapshot_init(self):
        tot = 0.0
        for pn, p in self.params:
            s = p.detach().float().cpu().clone()
            self.init_snaps[pn] = s
            tot += float(s.pow(2).sum())
        self.init_norm = math.sqrt(tot)


# ----------------------------------------------------------------------------- #
#  Capacity helpers (operate on singular values)
# ----------------------------------------------------------------------------- #
def _svdvals(W):
    W = W.detach().float().cpu()
    if W.ndim != 2:
        return None
    try:
        return torch.linalg.svdvals(W)
    except Exception:
        try:
            return torch.linalg.svdvals(W + 1e-8 * torch.randn_like(W))
        except Exception:
            return None


def _capacity_from_svals(s):
    if s is None or s.numel() == 0:
        return float('nan'), float('nan'), float('nan')
    s = s.clamp_min(0)
    fro2 = float((s ** 2).sum())
    spec2 = float((s.max()) ** 2)
    stable_rank = fro2 / (spec2 + 1e-12)
    p = (s ** 2)
    p = p / (p.sum() + 1e-12)
    p = p.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    eff_rank = math.exp(entropy)
    return stable_rank, entropy, eff_rank


# ----------------------------------------------------------------------------- #
#  Tracker
# ----------------------------------------------------------------------------- #
_METRICS = [
    'grad_param_norm',
    'grad_effective',
    'grad_inflow',
    'grad_outflow',
    'fwd_branch_ratio',
    'drift_rel_fro',
    'drift_cosine',
    'drift_norm_ratio',
    'cap_stable_rank',
    'cap_spectral_entropy',
    'cap_effective_rank',
]


class GradientFlowTracker:
    def __init__(self, model, log_dir, sample_frac=0.1, every=0, verbose=True):
        """
        model       : the nn.Module being trained (Exp_Main.model)
        log_dir     : directory for the per-metric CSVs
        sample_frac : auto cadence = max(1, int(sample_frac * steps_per_epoch))
        every       : if > 0, overrides the auto cadence (log every `every` iters)
        """
        self.model = model
        self.log_dir = log_dir
        self.sample_frac = sample_frac
        self.every = every if every and every > 0 else None
        self.verbose = verbose
        self.eps = 1e-12
        self.active = False
        self._hooks = []

        os.makedirs(log_dir, exist_ok=True)

        self.groups = self._discover(model)
        self.labels = [g.label for g in self.groups]            # column order
        self._by_container = {g.container_name: g
                              for g in self.groups if g.container_name is not None}

        self._register_hooks(model)
        self._open_writers()

        if verbose:
            print('[GradientFlowTracker] tracking %d modules:' % len(self.groups))
            print('   ' + ', '.join(self.labels))
            print('[GradientFlowTracker] logging to %s' % os.path.abspath(log_dir))

    # ----- discovery -------------------------------------------------------- #
    def _discover(self, model):
        tmp = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            res = classify(name)
            if res is None:
                continue
            label, container, sort_key = res
            g = tmp.get(label)
            if g is None:
                g = _Group(label, sort_key)
                tmp[label] = g
            g.params.append((name, p))
            if container is not None and g.container_name is None:
                g.container_name = container

        groups = sorted(tmp.values(), key=lambda g: g.sort_key)
        for g in groups:
            g.snapshot_init()
        return groups

    # ----- hooks ------------------------------------------------------------ #
    def _register_hooks(self, model):
        mods = dict(model.named_modules())
        for cname, g in self._by_container.items():
            mod = mods.get(cname)
            if mod is None:
                continue
            self._hooks.append(
                mod.register_full_backward_hook(self._make_bwd_hook(g)))
            self._hooks.append(
                mod.register_forward_hook(self._make_fwd_hook(g)))

    def _make_bwd_hook(self, g):
        def hook(module, grad_input, grad_output):
            if not self.active:
                return
            g.inflow = _first_norm(grad_output)
            g.outflow = _first_norm(grad_input)
        return hook

    def _make_fwd_hook(self, g):
        def hook(module, inp, out):
            if not self.active:
                return
            o = _first_tensor(out)
            i = _first_tensor(inp)
            if o is None or i is None:
                g.fwd_ratio = float('nan')
            else:
                ni = float(i.detach().float().norm())
                no = float(o.detach().float().norm())
                g.fwd_ratio = no / (ni + self.eps)
        return hook

    # ----- writers ---------------------------------------------------------- #
    def _open_writers(self):
        self._files = {}
        self._writers = {}
        header = ['epoch', 'global_step', 'iter', 'lr'] + self.labels
        for m in _METRICS:
            path = os.path.join(self.log_dir, m + '.csv')
            f = open(path, 'w', newline='')
            w = csv.writer(f)
            w.writerow(header)
            f.flush()
            self._files[m] = f
            self._writers[m] = w

    # ----- cadence ---------------------------------------------------------- #
    def should_log(self, epoch, it, steps_per_epoch):
        if self.every is None:
            self.every = max(1, int(self.sample_frac * steps_per_epoch))
        return (it % self.every == 0) or (it == steps_per_epoch - 1)

    def arm(self):
        self.active = True

    def disarm(self):
        self.active = False

    # ----- the per-step measurement ---------------------------------------- #
    def log_step(self, epoch, it, global_step, lr):
        """Call AFTER loss.backward() and BEFORE the next optimizer.zero_grad().
        (For AMP, call scaler.unscale_(optimizer) first so grads are unscaled.)"""
        rows = {m: [] for m in _METRICS}

        for g in self.groups:
            vals = self._measure_group(g)
            for m in _METRICS:
                rows[m].append(vals[m])

        prefix = [epoch, global_step, it, lr]
        for m in _METRICS:
            self._writers[m].writerow(prefix + rows[m])
            self._files[m].flush()

    def _measure_group(self, g):
        out = {m: float('nan') for m in _METRICS}

        # ---- parameter gradients & norms (in quadrature over the group) ----
        gsum, wsum, dsum, dot = 0.0, 0.0, 0.0, 0.0
        any_grad = False
        for pn, p in g.params:
            w = p.detach()
            wsum += float(w.float().pow(2).sum())
            if p.grad is not None:
                any_grad = True
                gsum += float(p.grad.detach().float().pow(2).sum())
            w0 = g.init_snaps.get(pn)
            if w0 is not None:
                wc = w.float().cpu()
                dsum += float((wc - w0).pow(2).sum())
                dot += float((wc * w0).sum())

        grad_norm = math.sqrt(gsum) if any_grad else float('nan')
        weight_norm = math.sqrt(wsum)
        out['grad_param_norm'] = grad_norm
        out['grad_effective'] = (grad_norm / (weight_norm + self.eps)) if any_grad else float('nan')

        out['drift_rel_fro'] = math.sqrt(dsum) / (g.init_norm + self.eps)
        out['drift_norm_ratio'] = weight_norm / (g.init_norm + self.eps)
        out['drift_cosine'] = dot / ((weight_norm * g.init_norm) + self.eps)

        # ---- activation-side (filled by hooks during the armed backward) ---
        out['grad_inflow'] = g.inflow
        out['grad_outflow'] = g.outflow
        out['fwd_branch_ratio'] = g.fwd_ratio

        # ---- capacity (average over the 2D weight matrices in the group) ---
        srs, ses, ers = [], [], []
        for pn, p in g.params:
            if p.detach().ndim == 2:
                sr, se, er = _capacity_from_svals(_svdvals(p))
                if not math.isnan(sr):
                    srs.append(sr); ses.append(se); ers.append(er)
        if srs:
            out['cap_stable_rank'] = sum(srs) / len(srs)
            out['cap_spectral_entropy'] = sum(ses) / len(ses)
            out['cap_effective_rank'] = sum(ers) / len(ers)

        return out

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------- #
#  small utilities
# ----------------------------------------------------------------------------- #
def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for e in x:
            if torch.is_tensor(e):
                return e
    return None


def _first_norm(x):
    t = _first_tensor(x)
    if t is None:
        return float('nan')
    return float(t.detach().float().norm())
