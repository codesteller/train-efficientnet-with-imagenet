MODEL_DIR="./train_results"
DATA_DIR="/data"
INDX="./index_file"
BATCHSIZE=48

# TF_XLA_FLAGS=--tf_xla_cpu_global_jit

horovodrun -np 8 bash ./scripts/bind.sh -- python3 main.py \
  --mode "train_and_eval" \
  --arch "efficientnet-b4" \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --use_xla \
  --augmenter_name autoaugment \
  --weight_init fan_out \
  --lr_decay cosine \
  --max_epochs 500 \
  --train_batch_size $BATCHSIZE \
  --eval_batch_size $BATCHSIZE \
  --log_steps 100 \
  --save_checkpoint_freq 20 \
  --lr_init 0.005 \
  --batch_norm syncbn \
  --mixup_alpha 0.2 \
  --weight_decay 5e-6 \
  --resume_checkpoint
