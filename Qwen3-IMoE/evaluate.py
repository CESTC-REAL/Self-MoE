"""Zero/few-shot downstream evaluation on AQuA / GSM8K / CMMLU / C-Eval.

Default mode: generates answers for a random subset of each benchmark, scores
them inline (accuracy per benchmark) and stores the raw predictions as JSON
under --output_dir.

--score_only mode: skips model loading / generation and re-scores existing
prediction file(s) — useful when only the answer extraction needs to change.

Examples:
    # generate + score
    python evaluate.py \
        --model_type imoe \
        --run_name qwen-imoe-0.6b-sft \
        --checkpoint_path outputs \
        --data_dir data \
        --ds_name aqua \
        --num_samples 200

    # re-score an existing prediction file
    python evaluate.py --run_name qwen-imoe-0.6b-sft --score_only \
        --prediction_files results/qwen-imoe-0.6b-sft.json
"""

import argparse
import glob
import json
import os
import random as rm
import re
from os import path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from common.model_loading import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream evaluation for Dense / CS-MoE (IMoE) models")
    # Model
    parser.add_argument("--model_type", type=str, default="imoe", choices=["bs", "imoe"],
                        help="bs: load with AutoModelForCausalLM; imoe: load the CS-MoE class")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Training run name to evaluate (required unless --score_only "
                             "with explicit --prediction_files)")
    parser.add_argument("--checkpoint_path", type=str, default="outputs",
                        help="Directory that contains <run_name>/checkpoint-* or the final model")
    parser.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--config", type=str, default=None,
                        help="IMoE config json; defaults to configs/qwen3_0_6_A1_7b.json")

    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Benchmark data root directory")
    parser.add_argument("--ds_name", type=str, default="aqua", choices=["aqua", "gsm8k", "cmmlu", "ceval"])

    # Generation / sampling
    parser.add_argument("--num_samples", type=int, default=200, help="Number of evaluated samples")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results")

    # Re-scoring mode
    parser.add_argument("--score_only", action="store_true",
                        help="Skip generation and only score existing prediction file(s)")
    parser.add_argument("--prediction_files", type=str, nargs="+", default=None,
                        help="Prediction JSON file(s) for --score_only; "
                             "defaults to <output_dir>/<run_name>.json")
    return parser.parse_args()


def extract_prediction(predict: str, eval_set: str):
    """Extract the answer from a raw model response.

    Multiple-choice benchmarks: the `boxed{X}` option letter; GSM8K: the
    `<answer>...</answer>` numeric tag.
    """
    if eval_set == "gsm8k":
        match = re.search(r"<answer>\s*([+-]?\d+\.?\d*)\s*</answer>", predict)
    else:
        match = re.search(r"boxed{([A-Z])}", predict)
    return match.group(1) if match else None


def report_accuracy(results, prefix=""):
    """Compute and print accuracy per benchmark; returns {eval_set: accuracy}."""
    acc = {}
    for eval_set in sorted(set(line["eval_set"] for line in results)):
        lines = [line for line in results if line["eval_set"] == eval_set]
        correct = sum(
            1 for line in lines
            if (pred := extract_prediction(line["predict"], eval_set)) is not None
            and pred == str(line["answer"])
        )
        acc[eval_set] = correct / len(lines) if lines else 0.0
    print(f"Accuracy [{prefix}]: " + ", ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    return acc


def load_dataset(args, ds_name=None):
    """Load one benchmark and return (records, instruction suffix)."""
    data_dir = args.data_dir
    ds_name = ds_name or args.ds_name
    print(f"Loading benchmark {ds_name}...")

    if ds_name == "aqua":
        data_file = path.join(data_dir, "AQuA/data/test.json")
        lines = json.load(open(data_file, "r"))
        prompt = "Please output the correct option after ####"
    elif ds_name == "gsm8k":
        data_file = path.join(data_dir, "gsm8k/main/test.parquet")
        lines = pd.read_parquet(data_file, columns=["prompt", "reward_model"]).to_dict(orient="records")
        lines = [{
            "question": line["prompt"][0]["content"],
            "answer": line["reward_model"]["ground_truth"],
        } for line in lines]
        prompt = "Please output the final numeric answer after ####"
    elif ds_name == "cmmlu":
        data_files = glob.glob(path.join(data_dir, "cmmlu/test/*.csv"))
        print(f"Found {len(data_files)} CMMLU test files")
        lines = pd.concat([pd.read_csv(file) for file in data_files]).to_dict(orient="records")
        lines = [{
            "question": f"{line['question']}\nA. {line['A']}\nB. {line['B']}\nC. {line['C']}\nD. {line['D']}",
            "answer": line["Answer"],
        } for line in lines]
        prompt = "Please output the correct option after ####"
    elif ds_name == "ceval":
        data_files = glob.glob(path.join(data_dir, "ceval/**/test-00000-of-00001.parquet"))
        print(f"Found {len(data_files)} C-Eval test files")
        lines = pd.concat([pd.read_parquet(file) for file in data_files]).to_dict(orient="records")
        lines = [{
            "question": f"{line['question']}\nA. {line['A']}\nB. {line['B']}\nC. {line['C']}\nD. {line['D']}",
            "answer": line["answer"],
        } for line in lines]
        prompt = "Please output the correct option after ####"
    else:
        raise ValueError(f"Unsupported benchmark: {ds_name}")

    print(f"Loaded {len(lines)} samples; example:\n{lines[0]}")
    return lines, prompt


def load_model_from_args(args):
    checkpoint = path.join(args.checkpoint_path, args.run_name)
    print(f"Loading model from {checkpoint}...")
    model = load_model(checkpoint, model_type=args.model_type, config_path=args.config,
                       device=args.device, eval_mode=False)
    print("Model loaded")
    return model


def main():
    args = parse_args()
    if not args.score_only and not args.run_name:
        raise SystemExit("--run_name is required unless --score_only with explicit --prediction_files")
    if args.score_only:
        if not args.prediction_files and not args.run_name:
            raise SystemExit("--score_only requires --prediction_files or --run_name")
    print(args)
    rm.seed(args.seed)

    if args.score_only:
        files = args.prediction_files or [path.join(args.output_dir, f"{args.run_name}.json")]
        for file in files:
            with open(file, "r") as f:
                results = json.load(f)
            report_accuracy(results, prefix=path.basename(file))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model_from_args(args)

    results = []
    for eval_set in ["aqua", "gsm8k", "cmmlu", "ceval"]:
        lines, prompt = load_dataset(args, eval_set)
        rm.shuffle(lines)

        print(f"\nEvaluating {args.num_samples} random samples on {eval_set}...")
        print("=" * 50)
        for line in tqdm(lines[: args.num_samples], total=args.num_samples, desc=eval_set):
            messages = [{"role": "user", "content": f"{line['question']}\n\n{prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text=text, return_tensors="pt")

            outputs = model.generate(
                **{k: v.to(model.device) for k, v in inputs.items()},
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

            generated_text = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]

            results.append({
                "eval_set": eval_set,
                "answer": line["answer"],
                "predict": generated_text.strip(),
            })

    print("\nEvaluation finished.")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = path.join(args.output_dir, f"{args.run_name}.json")
        print(f"Saving predictions to {output_file}")
        json.dump(results, open(output_file, "w"), indent=4, ensure_ascii=False)

    report_accuracy(results, prefix=args.run_name)


if __name__ == "__main__":
    main()
