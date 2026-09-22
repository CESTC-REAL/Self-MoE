"""Pre-train a dense Qwen3 baseline (`--train_type bs`) or the CS-MoE / IMoE
model (`--train_type imoe`) with the HuggingFace Trainer.

Example (single node, 4 GPUs):
    accelerate launch --config_file configs/accelerate/accelerate_ddp.yaml \
        --num_processes 4 \
        pretraining.py \
        --train_type imoe \
        --model_size 0.6B \
        --config configs/qwen3_0_6_A0_6b.json \
        --train_data "data/train/*.parquet" \
        --batch_size 2 --grad_accum 32
"""

import argparse
import os
from os import path

import datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Qwen3Config,
    Qwen3ForCausalLM,
    TrainingArguments,
)

from common.data_io import load_files
from common.trainer import MoETrainer
from modeling_qwen3_imoe import Qwen3InflateMoeForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-training for Dense / CS-MoE (IMoE) models")
    # Model
    parser.add_argument("--train_type", type=str, default="imoe", choices=["bs", "imoe"],
                        help="bs: dense Qwen3 baseline; imoe: CS-MoE (IMoE) model")
    parser.add_argument("--model_size", type=str, default="0.6B", choices=["0.6B", "1.7B", "4B"],
                        help="Model scale; selects the default IMoE config and names the run")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the IMoE config json; defaults to configs/ by model size")
    parser.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-0.6B",
                        help="HF model id or local path providing the Qwen3 tokenizer")

    # Data
    parser.add_argument("--train_data", type=str, nargs="+", required=True,
                        help="Glob pattern(s) of pre-training corpora (.parquet / .json / .jsonl), "
                             "each file must contain a text field")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing / logging
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to ./outputs/<run_name>")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    return parser.parse_args()


def build_model(args, config=None):
    if args.train_type == "bs":
        qwen_config = Qwen3Config.from_pretrained(f"Qwen/Qwen3-{args.model_size}")
        model = Qwen3ForCausalLM(config=qwen_config)
    else:
        model = Qwen3InflateMoeForCausalLM(config=config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def main():
    args = parse_args()
    print(args)

    default_configs = {
        "0.6B": "configs/qwen3_0_6_A0_6b.json",
        "1.7B": "configs/qwen3_1_7_A4b.json",
        "4B": "configs/qwen3_4_A8b.json",
    }
    run_name = args.run_name or f"qwen-{args.train_type}-{args.model_size.lower()}-pt"
    output_dir = args.output_dir or path.join("outputs", run_name)
    print(f"Run name: {run_name} | output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    train_dataset = load_files(args.train_data)
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x[args.text_column], truncation=True, padding=False,
                            max_length=args.max_length),
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print(f"Total training samples: {len(train_dataset)}")

    if args.train_type == "imoe":
        config = AutoConfig.from_pretrained(args.config or default_configs[args.model_size],
                                            trust_remote_code=True)
        model = build_model(args, config=config)
    else:
        model = build_model(args)

    precision_args = {}
    if args.precision == "fp16":
        precision_args["fp16"] = True
    elif args.precision == "bf16":
        precision_args["bf16"] = True

    train_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=path.join(output_dir, "logs"),
        logging_steps=args.logging_steps,
        logging_first_step=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="tensorboard",
        seed=args.seed,
        **precision_args,
    )

    trainer = MoETrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        output_dir=output_dir,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.save_pretrained(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
