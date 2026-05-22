import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import JaquaConfig


PROMPTS = [
    "Hello. Who are you?",
    "Explain photosynthesis in three short bullets.",
    "Write a Python function that checks whether a number is prime.",
    "Summarize this sentence: The launch was delayed because of high winds.",
    "Refuse an unsafe request politely.",
]


def _cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


def _looks_broken(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True
    if len(set(stripped)) < 5:
        return True
    if re.fullmatch(r"([^\w\s])\1{7,}", stripped):
        return True
    if len(re.findall(r"[A-Za-z0-9]", stripped)) < max(6, len(stripped) // 8):
        return True
    return False


def main() -> None:
    cfg = JaquaConfig()

    if not os.path.isdir(cfg.merged_dir):
        raise RuntimeError(f"Missing merged model directory: {cfg.merged_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.merged_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.merged_dir,
        torch_dtype=_cuda_dtype(),
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    failures = []
    os.makedirs(os.path.dirname(cfg.validation_log_path), exist_ok=True)
    with open(cfg.validation_log_path, "w", encoding="utf-8") as log:
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
            log.write(f"PROMPT: {prompt}\nRESPONSE: {response.strip()}\n\n")
            print(f"[validate] prompt: {prompt}")
            print(response.strip())
            print()

            if _looks_broken(response):
                failures.append(prompt)

    if failures:
        raise RuntimeError(
            f"Validation generated weak/empty responses for {len(failures)} prompt(s). "
            f"See {cfg.validation_log_path}"
        )

    print(f"[validate] completed -> {cfg.validation_log_path}")


if __name__ == "__main__":
    main()
