#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune LiquidAI/LFM2-VL-* on Japanese text-only multi-turn chat data (.jsonl version).
- content_male_ja を使用
- データセットが限られているため、全ての assistant のみ予測対象 (on-all)
- multi_modal_projector / vision_tower は凍結
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ================== Dataset ==================

class JaChatTextOnlyOnAllDataset(Dataset):
    def __init__(self, json_path, system_message="日本語で自然にリアクションして。", split_ratio=0.98, split="train", seed=42):
        super().__init__()
        self.system_message = system_message

        data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"⚠️ Skip malformed line: {e}")
                    continue

        cleaned = []
        for item in data:
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue

            # content_male_ja のみ使用
            conv_pairs = []
            for c in convs:
                if c.get("role") in ["user", "assistant"]:
                    text = c.get("content_male_ja", "").strip()
                    if not text:
                        continue
                    conv_pairs.append({"role": c["role"], "text": text})
            if len(conv_pairs) < 2:
                continue

            cleaned.append(conv_pairs)

        # train / eval split
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(cleaned), generator=g).tolist()
        cutoff = int(len(cleaned) * split_ratio)
        idxs = perm[:cutoff] if split == "train" else perm[cutoff:]
        self.items = [cleaned[i] for i in idxs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        conv = self.items[idx]

        chat = [{"role": "system", "content": [{"type": "text", "text": self.system_message}]}]
        for turn in conv:
            chat.append({"role": turn["role"], "content": [{"type": "text", "text": turn["text"]}]})
        return chat


# ================== Collator (on-all masking) ==================

def create_all_loss_collate_fn(processor):
    def collate_fn(samples):
        batch = processor.apply_chat_template(
            samples, tokenize=True, return_tensors="pt", padding=True
        )

        # ---- ここがポイント ----
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
        elif torch.is_tensor(batch):
            input_ids = batch
        elif isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
            input_ids = torch.stack(batch)
        else:
            raise RuntimeError(f"❌ Unexpected type from processor: {type(batch)}")

        # -------------------------

        attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            # userブロックをマスク（assistant以外）
            for j in range(len(ids) - 1):
                if ids[j] == im_start_id and ids[j + 1] != assistant_id:
                    end = next((k for k in range(j + 2, len(ids))
                                if ids[k] == im_start_id), len(ids))
                    labels[i, j:end] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


# ================== Freeze Vision ==================

def freeze_vlm_vision_parts(model):
    frozen_layers = []
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name or "vision_tower" in name:
            param.requires_grad = False
            frozen_layers.append(name)
    print(f"🧊 凍結レイヤー数: {len(frozen_layers)}")
    return model


# ================== Main ==================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="LiquidAI/LFM2-VL-450M")
    p.add_argument("--chat_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output_onall")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--system_message", type=str, default="日本語で自然にリアクションして。")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    print("📋 ===== Text-only On-All Training Configuration =====")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = freeze_vlm_vision_parts(model)

    # === Dataset ===
    train_ds = JaChatTextOnlyOnAllDataset(args.chat_json, system_message=args.system_message, split="train", seed=args.seed)
    eval_ds = JaChatTextOnlyOnAllDataset(args.chat_json, system_message=args.system_message, split="eval", seed=args.seed)

    print(f"✅ Train samples: {len(train_ds)}")
    print(f"✅ Eval  samples: {len(eval_ds)}")

    collate_fn = create_all_loss_collate_fn(processor)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=args.report_to,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    print("\n🚀 Starting On-All Fine-tuning...\n")
    trainer.train()
    print("🎉 Training completed!")

    print("💾 Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
