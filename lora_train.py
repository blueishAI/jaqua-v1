import json
import os
from itertools import islice
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import JaquaConfig


def _cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


def _clean_messages(messages: List[Dict]) -> List[Dict[str, str]]:
    cleaned = []
    for m in messages:
        role = m.get("role", "user").strip().lower()
        content = m.get("content", "").strip()
        if role == "system":
            role = "system"
        elif role == "assistant":
            role = "assistant"
        else:
            role = "user"
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def _bitagent_messages(example: Dict) -> List[Dict[str, str]]:
    raw = example.get("conversation")
    if not raw:
        return []
    conversation = json.loads(raw) if isinstance(raw, str) else raw
    messages = []
    for item in conversation:
        role = str(item.get("role", "user")).strip().lower()
        content = item.get("content", "")
        if role == "tool call":
            payload = content if isinstance(content, dict) else {"name": "web_search", "arguments": {"query": str(content)}}
            messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": str(content)})
        elif role == "system":
            messages.append({"role": "system", "content": str(content)})
        else:
            messages.append({"role": "user", "content": str(content)})
    return messages


def _tokenize_chat(tokenizer, messages: List[Dict], max_length: int) -> Dict[str, List[int]]:
    messages = _clean_messages(messages)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
    )
    labels = [-100] * len(input_ids)

    prefix: List[Dict[str, str]] = []
    prev_len = 0
    for message in messages:
        current = prefix + [message]
        current_ids = tokenizer.apply_chat_template(
            current,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
        )
        current_len = min(len(current_ids), len(input_ids))
        if message["role"] == "assistant" and current_len > prev_len:
            labels[prev_len:current_len] = input_ids[prev_len:current_len]
        prefix = current
        prev_len = current_len
        if prev_len >= len(input_ids):
            break

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _example_messages(example: Dict, cfg: JaquaConfig) -> List[Dict[str, str]]:
    if "conversation" in example:
        return _bitagent_messages(example)

    if cfg.messages_column in example and example[cfg.messages_column]:
        return example[cfg.messages_column]

    prompt = example.get(cfg.prompt_column)
    response = example.get(cfg.response_column)
    if prompt and response:
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]

    raise KeyError(
        "Dataset must contain a chat messages column or prompt/response columns. "
        f"Looked for messages='{cfg.messages_column}', prompt='{cfg.prompt_column}', "
        f"response='{cfg.response_column}'."
    )


def _load_training_dataset(cfg: JaquaConfig, tokenizer):
    dataset_ids = [item.strip() for item in cfg.dataset_id.split(",") if item.strip()]
    splits = [item.strip() for item in cfg.dataset_split.split(",") if item.strip()]
    if len(splits) == 1 and len(dataset_ids) > 1:
        splits = splits * len(dataset_ids)
    if len(dataset_ids) != len(splits):
        raise ValueError(f"Dataset/split mismatch: {dataset_ids} vs {splits}")

    datasets = []
    per_dataset_max = cfg.max_samples // len(dataset_ids) if cfg.max_samples > 0 and len(dataset_ids) > 1 else cfg.max_samples
    for dataset_id, split in zip(dataset_ids, splits):
        if per_dataset_max > 0:
            stream = load_dataset(dataset_id, split=split, streaming=True)
            ds = Dataset.from_list(list(islice(stream, per_dataset_max)))
        else:
            ds = load_dataset(dataset_id, split=split)
        ds = ds.map(
            lambda example: _tokenize_chat(tokenizer, _example_messages(example, cfg), cfg.seq_len),
            remove_columns=ds.column_names,
        )
        datasets.append(ds)
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def _collate_batch(tokenizer, examples: List[Dict]) -> Dict[str, torch.Tensor]:
    labels = [example["labels"] for example in examples]
    features = [{"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")

    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for row in labels:
        pad_len = max_len - len(row)
        padded_labels.append(row + [-100] * pad_len)
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def main() -> None:
    cfg = JaquaConfig()
    os.makedirs(cfg.adapter_dir, exist_ok=True)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if device != "cuda" and rank == 0:
        print("[lora] WARNING: CUDA not found. LoRA path is intended for Kaggle GPU.")
    if rank == 0:
        print(f"[lora] artifact={cfg.artifact_name}")
        print(f"[lora] base_model={cfg.base_model_id}")
        print(f"[lora] dataset={cfg.dataset_id} split={cfg.dataset_split}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = _cuda_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.train()

    ds = _load_training_dataset(cfg, tokenizer)
    ds = ds.filter(lambda row: any(label != -100 for label in row["labels"]) and len(row["input_ids"]) > 0)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        collate_fn=lambda examples: _collate_batch(tokenizer, examples),
    )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lora_lr, weight_decay=0.0)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while step < cfg.lora_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(loader, start=1):
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out.logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite LoRA loss at step {step}: {float(loss.detach().cpu())}")
            (loss / cfg.grad_accum_steps).backward()

            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0 and rank == 0:
                print(f"[lora] step={step}/{cfg.lora_steps} batch={i}/{len(loader)} loss={float(loss.detach().cpu()):.4f}")

            if step % cfg.save_every == 0 and rank == 0:
                ckpt = os.path.join(cfg.output_dir, "checkpoints", cfg.artifact_name, f"step-{step}")
                saver = model.module if hasattr(model, "module") else model
                saver.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"[lora] saved checkpoint: {ckpt}")

            if step >= cfg.lora_steps:
                break

    if rank == 0:
        saver = model.module if hasattr(model, "module") else model
        saver.save_pretrained(cfg.adapter_dir)
        tokenizer.save_pretrained(cfg.adapter_dir)
        print(f"[lora] training finished -> {cfg.adapter_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
