"""Shared checkpoint loading for Dense and CS-MoE (IMoE) models."""

from os import path

from transformers import AutoConfig, AutoModelForCausalLM

from modeling_qwen3_imoe import Qwen3InflateMoeForCausalLM

DEFAULT_IMOE_CONFIG = "configs/qwen3_0_6_A1_7b.json"


def load_model(checkpoint, model_type="imoe", config_path=None, device=None, eval_mode=True):
    """Load a trained checkpoint.

    Args:
        checkpoint: path to the model directory.
        model_type: "bs" loads a dense HF model via AutoModelForCausalLM;
            "imoe" loads the CS-MoE model (Qwen3InflateMoeForCausalLM).
        config_path: explicit config json for IMoE checkpoints. Resolution
            order: explicit path > config.json inside the checkpoint >
            ``DEFAULT_IMOE_CONFIG``.
        device: optionally move the model to this device.
        eval_mode: call ``model.eval()`` after loading.
    """
    if model_type == "bs":
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    elif model_type == "imoe":
        if config_path is None and path.isfile(path.join(checkpoint, "config.json")):
            config_path = checkpoint  # prefer the config saved with the checkpoint
        config = AutoConfig.from_pretrained(config_path or DEFAULT_IMOE_CONFIG,
                                            trust_remote_code=True)
        model = Qwen3InflateMoeForCausalLM.from_pretrained(
            checkpoint, config=config, trust_remote_code=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected 'bs' or 'imoe')")

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model
