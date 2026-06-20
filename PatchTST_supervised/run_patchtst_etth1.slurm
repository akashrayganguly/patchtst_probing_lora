#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR PatchTST SUPERVISED TRAINING ON ETTh1  (+ GRADIENT DIAGNOSTICS)
#
# PatchTST is a single-process training job — no FSDP, no torchrun.
# We run 4 sequential prediction horizons (96, 192, 336, 720) on one GPU.
#
# Gradient-flow / parameter-drift / capacity diagnostics are written to
#   $WORK_DIR/grad_logs/<setting>/<metric>.csv
# and can be plotted afterwards with:
#   python plot_grad_flow.py --log_dir grad_logs/<setting>
#
# USAGE:
#   sbatch run_patchtst_etth1.slurm
#===============================================================================

#SBATCH --job-name=patchtst-etth1
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1                      # 1 GPU is sufficient — PatchTST on ETTh1
                                          # is lightweight (7 channels, small model)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                 # A few workers for DataLoader
#SBATCH --mem=32G
#SBATCH --time=02-00:00:00                # ETTh1 × 4 horizons × 100 epochs ~ a few hours
#SBATCH --output=logs/patchtst_etth1_%j.log
#SBATCH --error=logs/patchtst_etth1_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    PatchTST SUPERVISED TRAINING — ETTh1  (with gradient diagnostics)"
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
# ETTh1.csv must be at  $WORK_DIR/dataset/ETTh1.csv
# Download from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
# (same link as in the PatchTST README, provided by Autoformer repo)
if [ ! -f "dataset/ETTh1.csv" ]; then
    echo "ERROR: dataset/ETTh1.csv not found."
    echo "Download it from the Autoformer Google Drive link in the PatchTST README"
    echo "and place it at: $WORK_DIR/dataset/ETTh1.csv"
    exit 1
fi
echo "Data file found: dataset/ETTh1.csv"
echo "=========================================================================="

#===============================================================================
# HYPERPARAMETERS  (directly from the official etth1.sh in the repo)
#
# Model: PatchTST/42  (seq_len=336 → N = (336-16)/8 + 2 = 42 patches)
# These match the paper's Table 3 ETTh1 numbers.
#
# Small model size (H=4, D=16, F=128) is intentional — the paper uses a reduced
# model for ETTh1/ETTh2 to avoid overfitting on this smaller dataset.
#===============================================================================
model_name=PatchTST
seq_len=336
random_seed=2021

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

# ---- Diagnostics -------------------------------------------------------------
# TRACK=1 enables gradient-flow / drift / capacity logging. CSVs land in
# grad_logs/<setting>/. Sampling cadence is ~10% of an epoch (TRACK_FRAC).
# For a FAST probe of the "attention does nothing" question, you usually do NOT
# need 100 epochs — set TRAIN_EPOCHS=5 below and read the first-epoch traces.
TRACK=1
TRACK_FRAC=0.1
TRAIN_EPOCHS=100
# -----------------------------------------------------------------------------

echo "Model:        $model_name"
echo "Look-back:    seq_len=$seq_len"
echo "Patch length: 16, stride 8  → ~42 patches"
echo "Enc layers:   3,  heads: 4,  d_model: 16,  d_ff: 128"
echo "Horizons:     96 192 336 720"
echo "Epochs:       $TRAIN_EPOCHS,  batch: 128,  lr: 0.0001"
echo "Tracking:     TRACK=$TRACK  sample_frac=$TRACK_FRAC"
echo "=========================================================================="

#===============================================================================
# TRAINING LOOP — 4 prediction horizons
#===============================================================================
for pred_len in 96 192 336 720; do

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
        --patch_len  16 \
        --stride     8 \
        --des        'Exp' \
        --train_epochs $TRAIN_EPOCHS \
        --itr        1 \
        --batch_size 128 \
        --learning_rate 0.0001 \
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
    echo "TRAINING FAILED — check logs/patchtst_etth1_${SLURM_JOB_ID}.error"
fi
echo "=========================================================================="

exit $EXIT_CODE
