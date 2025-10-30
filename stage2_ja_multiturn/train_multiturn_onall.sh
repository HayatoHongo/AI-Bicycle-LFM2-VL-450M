
torchrun --nproc_per_node=8  stage2_ja_multiturn/ja_train_multiturn_onall.py \
  --model_id YOUR_STAGE1_MODEL_ID_ON_HUGGINGFACE \
  --chat_json /workspace/stage2_ja_multiturn/data/Japanese_multiturn_soda_kaken_train_131152.jsonl \
  --output_dir /workspace/output \
  --epochs 10 \
  --lr 2e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --bf16 \
  --tf32 \
  --report_to wandb \
  --logging_steps 20 \
  --seed 42
