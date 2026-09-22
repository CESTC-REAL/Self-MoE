# Qwen3-IMoE: CS-MoE implementation on the Qwen3 backbone

Reference implementation of **CS-MoE** (Cross-layer Shared Mixture-of-Experts, a.k.a.
IMoE / Global Experts Sharing) as a HuggingFace Transformers drop-in on top of
Qwen3-MoE. See the [paper](https://arxiv.org/abs/2609.22199) (EMNLP 2026 Main) for
the method description and results.

## Architecture summary

- **Global shared expert pool** (`Qwen3MoeSharedExperts`): `num_shared_experts` (M)
  experts instantiated **once** and referenced by every layer.
- **Fixed Path**: each layer owns
  `(num_experts - num_shared_experts) / num_hidden_layers` **local independent
  experts**, activated for every token with a constant gate of 1.0 (routing-free).
- **Dynamic Path**: a layer-specific router (`Qwen3MoeTopKRouter`) routes every
  token to `top_k = num_experts_per_tok - num_local_experts` experts of the shared
  pool; gate weights are the renormalized Top-K scores.
- **Layer-wise auxiliary load-balancing loss** (Switch-Transformer style) computed
  on the shared-pool router logits of every layer, added to the LM loss with
  `router_aux_loss_coef` when `config.output_router_logits=True`.

## Repository layout

```
Qwen3-IMoE/
├── modeling_qwen3_imoe.py     # CS-MoE model implementation (Transformers drop-in)
├── common/                    # shared utilities
│   ├── trainer.py             # MoETrainer (Trainer + log.jsonl logging)
│   ├── model_loading.py       # checkpoint loading (dense / CS-MoE)
│   └── data_io.py             # glob resolution + parquet/json dispatch
├── pretraining.py             # pre-training entry point (dense baseline / IMoE)
├── finetuning.py              # SFT entry point
├── evaluate.py                # downstream evaluation + accuracy scoring (AQuA / GSM8K / CMMLU / C-Eval)
├── loss_eval.py               # LM loss / perplexity on a text file
├── inference.py               # generation demo / sanity check
├── configs/
│   ├── qwen3_0_6_A0_6b.json   # CS-MoE 0.6B-A0.6B (more scales under the same scheme)
│   ├── qwen3_0_6_A1_7b.json   # CS-MoE 0.6B-A1.7B
│   ├── qwen3_1_7_A4b.json     # CS-MoE 1.7B-A4B
│   ├── qwen3_4_A8b.json       # CS-MoE 4B-A8B
│   └── accelerate/            # accelerate (DDP / FSDP / DeepSpeed) launch configs
├── data_processing/           # corpus preprocessing utilities (.jsonl.zst etc.)
├── scripts/                   # ready-to-use launch scripts
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick sanity check

Build a model from a config, run a forward pass and report physical vs.
activated parameter counts:

```bash
python modeling_qwen3_imoe.py --config configs/qwen3_0_6_A0_6b.json
```

## Pre-training

```bash
# Dense baseline
TRAIN_DATA="data/train/*.parquet" MODEL_SIZE=1.7B bash scripts/pretrain_baseline.sh

# CS-MoE (IMoE)
TRAIN_DATA="data/train/*.parquet" CONFIG=configs/qwen3_0_6_A0_6b.json bash scripts/pretrain_imoe.sh
```

Data: one or more glob patterns of `.parquet` / `.json` / `.jsonl` files, each
containing a `text` field (override with `--text_column`). All optimization
hyperparameters (learning rate, warmup, batch size, precision, ...) are exposed
as CLI arguments — see `python pretraining.py --help`.

## Supervised fine-tuning

```bash
DATASET_PATH="data/sft/*.json" bash scripts/sft_imoe.sh
```

SFT corpora must provide either chat-style `messages` or `question`/`answer`
fields per sample; the loss is computed on the assistant response only.

## Evaluation

```bash
# Generate predictions and score them inline (random subset per benchmark)
python evaluate.py --model_type imoe --run_name qwen-imoe-0.6b-sft \
    --checkpoint_path outputs --data_dir data --ds_name aqua --num_samples 200

# Re-score an existing prediction file without regenerating
python evaluate.py --score_only --prediction_files results/qwen-imoe-0.6b-sft.json
```

Benchmarks and expected on-disk layout under `--data_dir`:

| Name | Description | Format |
|---|---|---|
| `aqua` | AQuA math word problems | `AQuA/data/test.json` |
| `gsm8k` | GSM8K | `gsm8k/main/test.parquet` |
| `cmmlu` | CMMLU | `cmmlu/test/*.csv` |
| `ceval` | C-Eval | `ceval/**/test-00000-of-00001.parquet` |

## Loss / perplexity and inference

```bash
# LM loss & PPL over a text file
python loss_eval.py --checkpoint outputs/qwen-imoe-0.6b-pt --model_type imoe --text_file corpus.txt

# Chat generation
python inference.py --checkpoint outputs/qwen-imoe-0.6b-sft --model_type imoe --prompt "你好"
```

## Loading checkpoints

Trained checkpoints can be loaded directly (the config ships
`auto_architectures`-free, class-explicit loading):

```python
from modeling_qwen3_imoe import Qwen3InflateMoeForCausalLM

model = Qwen3InflateMoeForCausalLM.from_pretrained("outputs/qwen-imoe-0.6b-pt")
```

The global shared pool is stored **once** per checkpoint (layer-side copies are
registered as tied weights and deduplicated at save time).

## Notes

- The 8B / 12B configurations used in the paper's full-scale runs are added to
  `configs/` analogously (`num_shared_experts` = 324 / 540 at `d_exp` 2,432).
- `configs/accelerate/` contains DDP / FSDP / DeepSpeed ZeRO-2 launch configs;
  `scripts/*.sh` pass `--num_processes` explicitly so the YAML value is ignored.
