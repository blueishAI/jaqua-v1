import os
from dataclasses import dataclass


@dataclass
class JaquaConfig:
    # Artifact identity
    param_label: str = os.getenv("JAQUA_PARAM_LABEL", "1.5b")
    variant: str = os.getenv("JAQUA_VARIANT", "base")

    # Hugging Face source
    base_model_id: str = os.getenv("JAQUA_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    # LoRA training schedule
    seq_len: int = int(os.getenv("JAQUA_SEQ_LEN", "192"))
    micro_batch_size: int = int(os.getenv("JAQUA_MICRO_BATCH", "1"))
    grad_accum_steps: int = int(os.getenv("JAQUA_GRAD_ACCUM", "1"))
    log_every: int = int(os.getenv("JAQUA_LOG_EVERY", "20"))
    save_every: int = int(os.getenv("JAQUA_SAVE_EVERY", "100"))
    lora_r: int = int(os.getenv("JAQUA_LORA_R", "16"))
    lora_alpha: int = int(os.getenv("JAQUA_LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("JAQUA_LORA_DROPOUT", "0.05"))
    lora_lr: float = float(os.getenv("JAQUA_LORA_LR", "2e-4"))
    lora_steps: int = int(os.getenv("JAQUA_LORA_STEPS", "1200"))

    # Data
    dataset_id: str = os.getenv("JAQUA_DATASET", "HuggingFaceH4/ultrachat_200k")
    dataset_split: str = os.getenv("JAQUA_DATASET_SPLIT", "train_sft")
    max_samples: int = int(os.getenv("JAQUA_MAX_SAMPLES", "100000"))
    messages_column: str = os.getenv("JAQUA_MESSAGES_COLUMN", "messages")
    prompt_column: str = os.getenv("JAQUA_PROMPT_COLUMN", "prompt")
    response_column: str = os.getenv("JAQUA_RESPONSE_COLUMN", "response")

    # Paths
    work_dir: str = os.getenv("JAQUA_WORK_DIR", "/kaggle/working")
    output_dir: str = os.getenv("JAQUA_OUTPUT_DIR", "/kaggle/working/output")
    hf_cache: str = os.getenv("HF_HOME", "/kaggle/temp/hf_cache")

    @property
    def artifact_name(self) -> str:
        return f"jaqua-{self.param_label}-{self.variant}"

    @property
    def adapter_dir(self) -> str:
        return os.path.join(self.output_dir, "adapters", self.artifact_name)

    @property
    def merged_dir(self) -> str:
        return os.path.join(self.output_dir, "merged", f"{self.artifact_name}-F16")

    @property
    def gguf_dir(self) -> str:
        return os.path.join(self.output_dir, "gguf")

    @property
    def validation_log_path(self) -> str:
        return os.path.join(self.output_dir, "logs", f"{self.artifact_name}-validation.log")
