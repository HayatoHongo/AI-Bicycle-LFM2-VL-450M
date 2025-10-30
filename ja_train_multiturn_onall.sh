
torchrun --nproc_per_node=8  script_and_data/ja_multiturn/ja_train_multiturn_onall.py \
  --model_id HayatoHongo/lfm2-vl-ja-finetuned-enmt1ep \
  --chat_json /workspace/script_and_data/ja_multiturn/data/Japanese_multiturn_soda_kaken_train_131152.jsonl \
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
