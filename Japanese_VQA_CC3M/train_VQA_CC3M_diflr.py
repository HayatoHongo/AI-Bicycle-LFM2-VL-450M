#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune LiquidAI/LFM2-VL-* on Japanese chat data (.jsonl version).
with multi learning rate for language vs vision modules.
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    set_seed,
)
from torch.optim import AdamW

from trl import SFTConfig, SFTTrainer


# ================== Dataset ==================

class JaChatImageDataset(Dataset):
    def __init__(self, json_path, image_root, system_message, split_ratio=0.98, split="train", seed=42):
        super().__init__()
        self.system_message = system_message
        self.image_root = image_root

        # --- load JSONL ---
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

        # --- clean ---
        cleaned = []
        for item in data:
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue
            human = next((c for c in convs if c.get("from") == "human"), None)
            gpt = next((c for c in convs if c.get("from") == "gpt"), None)
            if not human or not gpt:
                continue
            q_ja = human.get("value_ja", "")
            a_ja = gpt.get("value_ja", "")
            if not a_ja:
                continue
            img_rel = item.get("image")
            if not img_rel:
                continue
            img_path = os.path.join(image_root, img_rel)
            if not os.path.exists(img_path):
                continue
            cleaned.append({
                "image_path": img_path,
                "question_ja": q_ja,
                "answer_ja": a_ja,
            })

        # --- split ---
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(cleaned), generator=g).tolist()
        cutoff = int(len(cleaned) * split_ratio)
        if split == "train":
            idxs = perm[:cutoff]
        else:
            idxs = perm[cutoff:]
        self.items = [cleaned[i] for i in idxs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {row['image_path']} ({e})")

        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": row["question_ja"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": row["answer_ja"]}]},
        ]


# ================== Collator ==================

def create_masked_collate_fn(processor):
    def collate_fn(samples):
        batch = processor.apply_chat_template(
            samples, tokenize=True, return_tensors="pt", padding=True
        )
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch
            batch = {"input_ids": input_ids}
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            for j in range(len(ids) - 1):
                if ids[j] == im_start_id and ids[j + 1] == assistant_id:
                    labels[i, : j + 2] = -100
                    break

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        return batch
    return collate_fn


# ================== Utility: Multi LR ==================

def get_multi_lr_param_groups(model, base_lr, text_lr_scale=0.1):
    lang_params, vision_params, other_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "language_model" in name:
            lang_params.append(param)
        elif "multi_modal_projector" in name or "vision_tower" in name:
            vision_params.append(param)
        else:
            other_params.append(param)

    print(f"🧠 Language params: {len(lang_params)} tensors (lr ×{text_lr_scale})")
    print(f"👁️ Vision params: {len(vision_params)} tensors (lr ×1.0)")
    print(f"🪄 Other params: {len(other_params)} tensors (lr ×1.0)")

    return [
        {"params": vision_params, "lr": base_lr},
        {"params": lang_params, "lr": base_lr * text_lr_scale},
        {"params": other_params, "lr": base_lr},
    ]


# ================== Main ==================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="LiquidAI/LFM2-VL-450M")
    p.add_argument("--chat_json", type=str, required=True)
    p.add_argument("--image_folder", type=str, default="/workspace/images")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--text_lr_scale", type=float, default=0.1, help="Scale for language_model LR")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--run_name", type=str, default="lfm2-vl-fullft-ja")
    p.add_argument("--system_message", type=str, default="日本語で自然にリアクションして。")
    p.add_argument("--train_split_ratio", type=float, default=0.98)
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

    print("📋 ===== Training Configuration =====")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # ===== Processor / Model =====
    print("📚 Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, max_image_tokens=256)
    print("🧠 Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("🧩 Gradient checkpointing: ON")

    # ===== Datasets =====
    print("📦 Building datasets...")
    train_ds = JaChatImageDataset(
        json_path=args.chat_json,
        image_root=args.image_folder,
        system_message=args.system_message,
        split_ratio=args.train_split_ratio,
        split="train",
        seed=args.seed,
    )
    eval_ds = JaChatImageDataset(
        json_path=args.chat_json,
        image_root=args.image_folder,
        system_message=args.system_message,
        split_ratio=args.train_split_ratio,
        split="eval",
        seed=args.seed,
    )

    collate_fn = create_masked_collate_fn(processor)

    # ===== SFT Config =====
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
        report_to=(args.report_to if args.report_to != "none" else None),
        run_name=args.run_name,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
    )

    # ===== Optimizer with multi-LR =====
    param_groups = get_multi_lr_param_groups(model, args.lr, args.text_lr_scale)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # ===== Trainer =====
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        optimizers=(optimizer, None),
    )

    print("\n🚀 Starting SFT training with multi-LR...\n")
    trainer.train()
    print("🎉 SFT training completed!")

    print("💾 Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    print("✅ Done.")


if __name__ == "__main__":
    main()
