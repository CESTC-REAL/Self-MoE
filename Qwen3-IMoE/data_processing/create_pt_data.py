"""Utilities for reading compressed JSONL corpora (.jsonl.zst) used in
pre-training data preparation.

Example:
    python -m data_processing.create_pt_data --input corpus.jsonl.zst --output corpus.jsonl
"""

import argparse
import io
import json

import zstandard as zstd


def read_jsonl_zst(file_path):
    """Read a .jsonl.zst file and return the list of parsed records."""
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            return [json.loads(line) for line in text_stream if line.strip()]


def decompress_to_jsonl(input_path, output_path):
    """Decompress a .jsonl.zst file into a plain .jsonl file."""
    n = 0
    with open(input_path, "rb") as fh, open(output_path, "w", encoding="utf-8") as out:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    print(f"Wrote {n} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress .jsonl.zst corpora to .jsonl")
    parser.add_argument("--input", type=str, required=True, help="Input .jsonl.zst file")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    args = parser.parse_args()
    decompress_to_jsonl(args.input, args.output)
