if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=PatchTST

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021

# ---- Diagnostics knobs -------------------------------------------------------
# TRACK=1 enables gradient-flow / drift / capacity logging into grad_logs/.
# TRACK_FRAC ~ fraction of an epoch between samples (ILI epochs are tiny, so
# this often collapses to "every step" — cheap and fine here).
TRACK=1
TRACK_FRAC=0.1
# -----------------------------------------------------------------------------

# ---- Low-rank head -----------------------------------------------------------
# HEAD_RANK = r for the LoRA-style forecasting head  W (N x H) = A (N x r) @ B (r x H).
# Set HEAD_RANK<=0 to fall back to the original full-rank head (baseline run).
HEAD_RANK=5
export PATCHTST_HEAD_RANK=$HEAD_RANK   # read by models/PatchTST.py (no CLI flag needed)
# -----------------------------------------------------------------------------

for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --track_gradients $TRACK \
      --track_log_dir ./grad_logs \
      --track_sample_frac $TRACK_FRAC \
      --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
