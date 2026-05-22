import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import JaquaConfig


def _cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


def main() -> None:
    cfg = JaquaConfig()

    if not os.path.isdir(cfg.adapter_dir):
        raise RuntimeError(f"Missing LoRA adapter directory: {cfg.adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=_cuda_dtype(),
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    model = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(cfg.adapter_dir, use_fast=True)
    os.makedirs(cfg.merged_dir, exist_ok=True)
    model.save_pretrained(cfg.merged_dir, safe_serialization=False)
    tok.save_pretrained(cfg.merged_dir)
    print(f"[lora] merged model saved -> {cfg.merged_dir}")


if __name__ == "__main__":
    main()
