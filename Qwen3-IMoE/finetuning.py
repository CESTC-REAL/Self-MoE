"""Supervised fine-tuning (SFT) for a Dense or CS-MoE (IMoE) checkpoint.

Expected SFT data: one or more .json / .jsonl / .parquet files where each
sample either contains `messages` (chat format) or `question`/`answer` fields.
The loss is computed on the assistant response only.

Example:
    accelerate launch --config_file configs/accelerate/accelerate_ddp.yaml \
        --num_processes 4 \
        finetuning.py \
        --train_type imoe \
        --model_size 0.6B \
        --model_path outputs/qwen-imoe-0.6b-pt \
        --dataset_path "data/sft/*.json" \
        --batch_size 2 --grad_accum 8
"""

import argparse
from functools import partial
from os import path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from common.data_io import load_files
from common.trainer import MoETrainer
from modeling_qwen3_imoe import Qwen3InflateMoeForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="SFT for Dense / CS-MoE (IMoE) models")
    # Model
    parser.add_argument("--train_type", type=str, default="imoe", choices=["bs", "imoe"],
                        help="Which kind of checkpoint to start from")
    parser.add_argument("--model_size", type=str, default="0.6B", choices=["0.6B", "1.7B", "4B"],
                        help="Used for default run/checkpoint naming only")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Checkpoint to fine-tune; defaults to ./outputs/qwen-<type>-<size>-pt")
    parser.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-0.6B")

    # Data
    parser.add_argument("--dataset_path", type=str, nargs="+", required=True,
                        help="Glob pattern(s) of SFT corpora (.json / .jsonl / .parquet)")
    parser.add_argument("--cache_dir", type=str, default=None)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5000)
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
    return parser.parse_args()


def to_question_answer(example):
    """Normalize a sample to {question, answer}."""
    if "messages" in example:
        return {
            "question": example["messages"][0]["content"],
            "answer": example["messages"][1]["content"],
        }
    return {"question": example["question"], "answer": example["answer"]}


def load_sft_data(patterns, seed=42):
    ds = load_files(patterns)
    ds = ds.map(to_question_answer, remove_columns=ds.column_names)
    return ds.shuffle(seed=seed)


def preprocess_function(example, tokenizer, max_length: Optional[int] = 4096):
    messages = [{
        "role": "user",
        "content": example["question"],
    }, {
        "role": "assistant",
        "content": example["answer"],
    }]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = tokenized["input_ids"]

    # Mask everything before (and including) the response marker so that the
    # loss is computed on the assistant answer only.
    indices = torch.where(input_ids == tokenizer.encode("assistant")[0])[-1].tolist()
    labels = input_ids.clone()
    labels[:, : indices[-1] + 1] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def collect_fn(batch, tokenizer, max_length: Optional[int] = 3072):
    batch = {
        "input_ids": [torch.Tensor(example["input_ids"][:max_length]).to(torch.long) for example in batch],
        "attention_mask": [torch.Tensor(example["attention_mask"][:max_length]).to(torch.bool) for example in batch],
        "labels": [torch.Tensor(example["labels"][:max_length]).to(torch.long) for example in batch],
    }
    max_len = max(input_ids.shape[-1] for input_ids in batch["input_ids"])
    return {
        "input_ids": torch.cat(
            [F.pad(x, (0, max_len - x.shape[-1]), "constant", tokenizer.pad_token_id) for x in batch["input_ids"]],
            dim=0,
        ),
        "attention_mask": torch.cat(
            [F.pad(x, (0, max_len - x.shape[-1]), "constant", False) for x in batch["attention_mask"]],
            dim=0,
        ),
        "labels": torch.cat(
            [F.pad(x, (0, max_len - x.shape[-1]), "constant", -100) for x in batch["labels"]],
            dim=0,
        ),
    }


def main():
    args = parse_args()
    print(args)

    run_name = args.run_name or f"qwen-{args.train_type}-{args.model_size.lower()}-sft"
    output_dir = args.output_dir or path.join("outputs", run_name)
    model_path = args.model_path or path.join("outputs", f"qwen-{args.train_type}-{args.model_size.lower()}-pt")
    print(f"Run name: {run_name} | output: {output_dir} | init from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    train_dataset = load_sft_data(args.dataset_path)
    print(f"Total SFT samples: {len(train_dataset)}")
    train_dataset = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer, max_length=args.max_length),
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Building SFT labels",
    )
    print(f"Tokenized sample: {train_dataset[0]['labels'][-5:]}")

    if args.train_type == "imoe":
        model = Qwen3InflateMoeForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

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
        lr_scheduler_type="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="tensorboard",
        seed=args.seed,
        **precision_args,
    )

    trainer = MoETrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=partial(collect_fn, tokenizer=tokenizer, max_length=args.max_length),
        output_dir=output_dir,
    )

    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
