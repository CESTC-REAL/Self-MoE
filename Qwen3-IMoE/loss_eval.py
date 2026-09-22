"""Compute the LM loss / perplexity of a checkpoint over a text corpus,
token-budget weighted across fixed-length windows.

Example:
    python loss_eval.py --checkpoint outputs/qwen-imoe-0.6b-pt \
        --model_type imoe --text_file corpus.txt --device cuda:0
"""

import argparse

import torch
from transformers import AutoTokenizer

from common.model_loading import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="LM loss / perplexity evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="imoe", choices=["bs", "imoe"])
    parser.add_argument("--config", type=str, default=None,
                        help="IMoE config json; defaults to configs/qwen3_0_6_A1_7b.json")
    parser.add_argument("--tokenizer_id", type=str, default=None,
                        help="Defaults to the checkpoint itself")
    parser.add_argument("--text_file", type=str, required=True, help="UTF-8 text file to evaluate")
    parser.add_argument("--max_length", type=int, default=1024, help="Window length")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of windows per forward")
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Optionally cap the number of evaluated windows")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.checkpoint, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_model(args.checkpoint, model_type=args.model_type, config_path=args.config,
                       device=args.device)

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    n_windows = (len(token_ids) - 1) // args.max_length
    if args.max_windows is not None:
        n_windows = min(n_windows, args.max_windows)
    if n_windows <= 0:
        raise ValueError("Text too short for one window; lower --max_length or provide more text")
    print(f"{len(token_ids)} tokens -> {n_windows} windows of {args.max_length}")

    total_nll, total_tokens = 0.0, 0
    for start in range(0, n_windows * args.max_length, args.batch_size * args.max_length):
        window_ids = []
        for w in range(start // args.max_length, min((start // args.max_length) + args.batch_size, n_windows)):
            window_ids.append(token_ids[w * args.max_length:(w + 1) * args.max_length])
        input_ids = torch.stack(window_ids).to(model.device)
        labels = input_ids.clone()
        loss = model(input_ids=input_ids, labels=labels).loss
        n_tokens = input_ids.numel() - input_ids.shape[0]  # each window predicts len-1 tokens
        total_nll += loss.item() * n_tokens
        total_tokens += n_tokens
        print(f"windows [{start // args.max_length}, "
              f"{start // args.max_length + len(window_ids)}) | loss {loss.item():.4f}")

    avg_nll = total_nll / total_tokens
    print(f"\nTokens evaluated: {total_tokens}")
    print(f"Loss: {avg_nll:.4f} | Perplexity: {torch.exp(torch.tensor(avg_nll)).item():.4f}")


if __name__ == "__main__":
    main()
