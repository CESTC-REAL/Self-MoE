"""Quick inference demo / sanity check for a Dense or CS-MoE (IMoE) checkpoint.

Examples:
    # chat generation
    python inference.py --checkpoint outputs/qwen-imoe-0.6b-sft --model_type imoe \
        --prompt "介绍一下大语言模型中的混合专家架构" --max_new_tokens 512

    # teacher-forced loss on a piece of text
    python inference.py --checkpoint outputs/qwen-imoe-0.6b-sft --model_type imoe \
        --eval_text_file eval_text.txt
"""

import argparse

import torch
from transformers import AutoTokenizer

from common.model_loading import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference demo for Dense / CS-MoE (IMoE) models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_type", type=str, default="imoe", choices=["bs", "imoe"])
    parser.add_argument("--config", type=str, default=None,
                        help="IMoE config json; defaults to configs/qwen3_0_6_A1_7b.json")
    parser.add_argument("--tokenizer_id", type=str, default=None,
                        help="Defaults to the checkpoint itself")
    # Generation
    parser.add_argument("--prompt", type=str, default=None, help="User message for chat generation")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Or: teacher-forced loss on a text file
    parser.add_argument("--eval_text_file", type=str, default=None,
                        help="Compute the LM loss on this text file instead of generating")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.checkpoint, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_model(args.checkpoint, model_type=args.model_type, config_path=args.config,
                       device=args.device)

    if args.eval_text_file:
        # Teacher-forced LM loss on the given text.
        with open(args.eval_text_file, "r", encoding="utf-8") as f:
            text = f.read()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)
        inputs["labels"] = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"Loss: {outputs.loss.item():.4f} | Perplexity: {torch.exp(outputs.loss).item():.4f}")
        return

    if not args.prompt:
        raise ValueError("Provide --prompt for generation or --eval_text_file for loss evaluation")

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    response = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False
    )[0]
    print(response)


if __name__ == "__main__":
    main()
