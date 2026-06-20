#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR PatchTST SUPERVISED TRAINING ON ILI  (+ GRADIENT DIAGNOSTICS)
#
# national_illness.csv (ILI) — 7 channels, custom 0.7/0.2 split loader.
# PatchTST is a single-process training job — no FSDP, no torchrun.
# We run 4 sequential prediction horizons (24, 36, 48, 60) on one GPU.
#
# Gradient-flow / parameter-drift / capacity diagnostics are written to
#   $WORK_DIR/grad_logs/<setting>/<metric>.csv
# and can be plotted afterwards with:
#   python plot_grad_flow.py --log_dir grad_logs/<setting>
#
# USAGE:
#   sbatch run_patchtst_ili.slurm
#===============================================================================

#SBATCH --job-name=patchtst-ili
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1                      # 1 GPU is plenty — ILI is tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                 # A few workers for DataLoader
#SBATCH --mem=32G
#SBATCH --time=02-00:00:00                # ILI x 4 horizons x 100 epochs ~ minutes
#SBATCH --output=logs/patchtst_ili_%j.log
#SBATCH --error=logs/patchtst_ili_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    PatchTST SUPERVISED TRAINING — ILI  (with gradient diagnostics)"
echo "=========================================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "=========================================================================="

#===============================================================================
# SETUP
#===============================================================================
mkdir -p logs/LongForecasting

# ---- Edit this path to wherever you cloned yuqinie98/PatchTST ---------------
WORK_DIR="/home/akashg/time-series/python-files/patchtst_probing/PatchTST_supervised"
# -----------------------------------------------------------------------------

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
echo "Working directory: $(pwd)"

# ---- Conda environment -------------------------------------------------------
source /home/akashg/miniconda3/etc/profile.d/conda.sh
conda activate informerworking          # reuse your existing environment; it
                                        # already has PyTorch. If deps are missing
                                        # run:  pip install -r requirements.txt

echo "Python:  $(which python)"
echo "Version: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================================================="

# ---- Data --------------------------------------------------------------------
# national_illness.csv must be at  $WORK_DIR/dataset/national_illness.csv
# Download from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
# (same link as in the PatchTST README, provided by Autoformer repo)
if [ ! -f "dataset/national_illness.csv" ]; then
    echo "ERROR: dataset/national_illness.csv not found."
    echo "Download it from the Autoformer Google Drive link in the PatchTST README"
    echo "and place it at: $WORK_DIR/dataset/national_illness.csv"
    exit 1
fi
echo "Data file found: dataset/national_illness.csv"
echo "=========================================================================="

#===============================================================================
# HYPERPARAMETERS  (directly from the official illness.sh in the repo)
#
# Model: PatchTST  (seq_len=104, patch_len=24, stride=2
#                   -> N = (104-24)/2 + 1 = 41, +1 end-pad = 42 patches)
# These match the paper's ILI numbers.
#
# Small model size (H=4, D=16, F=128) is intentional — ILI is a very small
# dataset (~966 rows), so a reduced model avoids overfitting. lradj=constant.
#===============================================================================
model_name=PatchTST
seq_len=104
random_seed=2021

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

# ---- Diagnostics -------------------------------------------------------------
# TRACK=1 enables gradient-flow / drift / capacity logging. CSVs land in
# grad_logs/<setting>/. Sampling cadence is ~10% of an epoch (TRACK_FRAC).
# ILI epochs are very short (few steps each), so TRACK_FRAC=0.1 may collapse to
# "every step" — that's fine and cheap here. For a fast probe of the
# "attention does nothing" question you usually do NOT need 100 epochs — set
# TRAIN_EPOCHS=5 below and read the first-epoch traces.
TRACK=1
TRACK_FRAC=0.1
TRAIN_EPOCHS=100

# ---- Low-rank head -----------------------------------------------------------
# HEAD_RANK = r for the LoRA-style forecasting head  W (N x H) = A (N x r) @ B (r x H).
# The ILI head is nn.Linear(head_nf=672, pred_len), i.e. W is [pred_len, 672].
# Set HEAD_RANK<=0 to fall back to the original full-rank head (baseline run).
HEAD_RANK=5
export PATCHTST_HEAD_RANK=$HEAD_RANK   # read by models/PatchTST.py (no CLI flag needed)
# -----------------------------------------------------------------------------

echo "Model:        $model_name"
echo "Look-back:    seq_len=$seq_len"
echo "Patch length: 24, stride 2  -> ~42 patches"
echo "Enc layers:   3,  heads: 4,  d_model: 16,  d_ff: 128"
echo "Horizons:     24 36 48 60"
echo "Epochs:       $TRAIN_EPOCHS,  batch: 16,  lr: 0.0025,  lradj: constant"
echo "Head rank:    HEAD_RANK=$HEAD_RANK  (<=0 = full-rank head)"
echo "Tracking:     TRACK=$TRACK  sample_frac=$TRACK_FRAC"
echo "=========================================================================="

#===============================================================================
# TRAINING LOOP — 4 prediction horizons
#===============================================================================
for pred_len in 24 36 48 60; do

    echo ""
    echo ">>> Starting pred_len=$pred_len  at $(date)"
    echo "------------------------------------------------------------"

    python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path  $root_path_name \
        --data_path  $data_path_name \
        --model_id   ${model_id_name}_${seq_len}_${pred_len} \
        --model      $model_name \
        --data       $data_name \
        --features   M \
        --seq_len    $seq_len \
        --pred_len   $pred_len \
        --enc_in     7 \
        --e_layers   3 \
        --n_heads    4 \
        --d_model    16 \
        --d_ff       128 \
        --dropout    0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len  24 \
        --stride     2 \
        --des        'Exp' \
        --train_epochs $TRAIN_EPOCHS \
        --lradj      'constant' \
        --itr        1 \
        --batch_size 16 \
        --learning_rate 0.0025 \
        --track_gradients $TRACK \
        --track_log_dir ./grad_logs \
        --track_sample_frac $TRACK_FRAC \
        2>&1 | tee logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log

    echo ">>> Finished pred_len=$pred_len  at $(date)"

done

#===============================================================================
# SUMMARY
#===============================================================================
EXIT_CODE=$?
echo ""
echo "=========================================================================="
echo "JOB SUMMARY"
echo "=========================================================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "ALL 4 HORIZONS COMPLETED SUCCESSFULLY"
    echo ""
    echo "Results are appended to:  $WORK_DIR/result.txt"
    echo "Per-run logs:             $WORK_DIR/logs/LongForecasting/"
    echo "Checkpoints:              $WORK_DIR/checkpoints/"
    echo "Diagnostic CSVs:          $WORK_DIR/grad_logs/<setting>/"
    echo ""
    echo "Plot with:  python plot_grad_flow.py --log_dir grad_logs/<setting>"
else
    echo "TRAINING FAILED — check logs/patchtst_ili_${SLURM_JOB_ID}.error"
fi
echo "=========================================================================="

exit $EXIT_CODE
