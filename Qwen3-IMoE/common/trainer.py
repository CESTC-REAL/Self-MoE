"""Trainer shared by pretraining.py and finetuning.py."""

import json
import os
from os import path
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Trainer


class MoETrainer(Trainer):
    """Trainer that additionally appends every log record to ``<output_dir>/log.jsonl``."""

    def __init__(self, *args, output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = open(path.join(output_dir, "log.jsonl"), "w")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        super().log(logs, start_time)
        logs = {"lr": self.lr_scheduler.get_last_lr()[0], **logs}
        self.log_file.write(json.dumps(logs) + "\n")
        self.log_file.flush()
