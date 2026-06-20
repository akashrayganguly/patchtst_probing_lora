# PatchTST gradient-flow & capacity diagnostics

Instrumentation added to answer: *"ablating attention/FFN barely changes the
loss — are gradients not reaching them?"*

The short version of the analysis: on ETTh1 that symptom is **usually not a
gradient-flow failure**. PatchTST on ETT/weather is dominated by RevIN + the
linear flatten head; the transformer body is close to functionally redundant
(the DLinear critique). Gradients can flow perfectly while the body does little.
So the instrumentation measures three *separate* things and lets the data decide.

---

## What was changed

New files
- `utils/grad_tracker.py` — the tracker (discovery, hooks, metrics, CSV writers).
- `plot_grad_flow.py` — turns the CSVs into the two plot views.
- `selftest_tracker.py` — run once on the HPC to confirm wiring before the big job.

Modified files
- `exp/exp_main.py` — creates a `GradientFlowTracker` in `train()`, arms hooks on
  sampled steps, calls `log_step()` right after `loss.backward()` (and
  `scaler.unscale_()` first under AMP), closes the tracker at the end.
- `run_longExp.py` — four new flags: `--track_gradients`, `--track_log_dir`,
  `--track_every`, `--track_sample_frac`.
- `run_patchtst_etth1.slurm` — passes `--track_gradients 1` and exposes
  `TRACK`, `TRACK_FRAC`, `TRAIN_EPOCHS` knobs.

Nothing about the model, optimisation, or results changes when
`--track_gradients 0` (the default). Overhead only occurs on sampled steps.

---

## The three questions, and which metric answers each

| Question | Metric(s) | Read it as |
|---|---|---|
| Is a learning signal *reaching* the module? | `grad_inflow` (‖∂L/∂out‖), `grad_outflow` (‖∂L/∂in‖) | near-zero inflow ⇒ no signal arrives — a genuine flow problem |
| Do its *weights actually move*? | `grad_effective` (‖∂L/∂W‖/‖W‖), `drift_rel_fro` (‖W−W₀‖/‖W₀‖), `drift_cosine`, `drift_norm_ratio` | grad present but drift≈0 ⇒ optimiser isn't moving it (LR/clipping/normalisation) |
| Does it *matter to the forward pass*? | `fwd_branch_ratio` (‖sublayer_out‖/‖residual_in‖) | tiny ratio ⇒ branch barely perturbs the residual stream ⇒ ablation won't change loss even though it trains fine |
| Is it *expressive / using its rank*? | `cap_stable_rank`, `cap_spectral_entropy`, `cap_effective_rank` | collapsing to ~1 ⇒ module has degenerated to near rank-1 (little useful structure) |

`grad_param_norm` is the raw ‖∂L/∂W‖ (in quadrature over the module's weight
tensors). `lr` is logged every row because OneCycleLR sweeps the learning rate —
`grad_effective` is only comparable across steps once you account for it.

### Metric definitions
- Drift: `‖W−W₀‖_F / ‖W₀‖_F`, `cos(W,W₀)`, `‖W‖/‖W₀‖`. `W₀` = the weights at
  initialisation, snapshotted when the tracker is built.
- Stable rank: `‖W‖_F² / ‖W‖_2²` = `Σσᵢ² / σ_max²`.
- Spectral entropy: `H = −Σ pᵢ log pᵢ` with `pᵢ = σᵢ² / Σσⱼ²` (energy-normalised).
- Effective rank: `exp(H)`.
- Inflow/outflow come from `register_full_backward_hook`; branch ratio from a
  forward hook. Embeddings that are bare `Parameter`s (`W_pos`, RevIN affine)
  have no activation hooks, so their inflow/outflow/branch columns are `NaN` —
  the parameter-side metrics still apply.

---

## Why some of your original plan was changed

- **KLD over weight rows → dropped.** Weights are signed; turning rows into
  probabilities is arbitrary and uninterpretable. Replaced by relative Frobenius
  drift (primary) + cosine + norm ratio. If you want a *distributional* change
  measure, the principled place to put it is the **singular-value spectrum**
  (rotation-invariant) — which is exactly what `cap_spectral_entropy` already
  tracks.
- **"Information capacity" relabelled.** Deviation-from-init measures *learning*,
  not capacity. Capacity/expressivity is the stable-rank / spectral-entropy
  family. Both are kept, just named correctly.
- **Inflow/outflow defined as activation gradients**, not parameter gradients —
  that is what actually "flows" in backprop and what diagnoses a flow failure.
- **Added the forward branch-contribution ratio**, which your plan didn't have
  and which is the single most likely explanation for your symptom.

---

## How to run

1. Sanity check the wiring (once, in the repo dir, env active):
   ```bash
   python selftest_tracker.py        # expects RESULT: PASS
   ```
2. Submit the job (tracking already on in the slurm):
   ```bash
   sbatch run_patchtst_etth1.slurm
   ```
   For a fast probe you rarely need 100 epochs — set `TRAIN_EPOCHS=5` in the
   slurm and read the first-epoch traces.
3. Plot:
   ```bash
   python plot_grad_flow.py --log_dir grad_logs/<setting>
   # e.g. grad_logs/ETTh1_336_96_PatchTST_ETTh1_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0
   ```
   PNGs land in `grad_logs/<setting>/plots/`: per metric a `__trace.png`
   (lines, x = global_step, one line per module) and a `__snapshot.png`
   (bars over modules at one step).

CSV layout (every metric file): columns
`epoch, global_step, iter, lr, <module_1>, <module_2>, ...`; one row per
sampled step. A row = across-architecture snapshot; a column = across-iteration
trace. Pivoting in pandas is trivial since the format is already wide.

---

## Decision tree for your symptom

```
ablating attn/ffn ≈ no loss change
│
├─ grad_inflow at attn/ffn ≈ 0 ?
│     YES → genuine flow failure. Check: residual path dominating, a detach,
│           dropout=0.3 wiping the branch, or norm placement. Rare here.
│     NO  → signal is arriving; go on.
│
├─ drift_rel_fro for attn/ffn stays ≈ 0 over training ?
│     YES → grads arrive but weights don't move. Suspect LR too low for the
│           body, OneCycle schedule, or BatchNorm in the block swamping updates.
│     NO  → weights are moving; go on.
│
└─ fwd_branch_ratio for attn/ffn is small (e.g. <0.1) ?
      YES → EXPECTED OUTCOME. The block trains but contributes little to the
            residual stream, so the linear head + RevIN already solve ETTh1.
            This is the DLinear story, not a bug. Confirm with an explicit
            ablation (zero the branch) and compare test MSE.
```

If `cap_stable_rank`/`cap_effective_rank` for attention collapse toward 1 while
the head stays high-rank, that's further evidence the body has degenerated to a
near-identity contribution rather than failing to receive gradients.
