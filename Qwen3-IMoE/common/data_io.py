"""Glob resolution and file-format dispatch shared by all data-loading scripts."""

import glob
from os import path

from datasets import concatenate_datasets, load_dataset


def resolve_files(patterns):
    """Expand glob pattern(s) into a sorted, de-duplicated list of files."""
    files = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No files matched: {pattern}")
        files.extend(matched)
    return sorted(set(files))


def load_files(patterns):
    """Load one or more .parquet / .json / .jsonl files into a single dataset."""
    parts = []
    for file in resolve_files(patterns):
        ext = path.splitext(file)[1].lower()
        if ext == ".parquet":
            ds = load_dataset("parquet", data_files=[file], split="train")
        elif ext in (".json", ".jsonl"):
            ds = load_dataset("json", data_files=[file], split="train")
        else:
            raise ValueError(f"Unsupported data format '{ext}' for {file}")
        print(f"Loaded {len(ds)} samples from {file}")
        parts.append(ds)
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)
