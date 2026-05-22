#!/usr/bin/env bash
set -euo pipefail

if [[ "${JAQUA_VERBOSE:-0}" == "1" ]]; then
  set -x
  export JAQUA_LOG_EVERY="${JAQUA_LOG_EVERY:-1}"
fi

cd "$(dirname "$0")"

export JAQUA_WORK_DIR="${JAQUA_WORK_DIR:-/kaggle/working}"
export JAQUA_OUTPUT_DIR="${JAQUA_OUTPUT_DIR:-/kaggle/working/output}"
export HF_HOME="${HF_HOME:-/kaggle/temp/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1
export USE_TF=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "${JAQUA_OUTPUT_DIR}/logs" "${JAQUA_OUTPUT_DIR}/gguf" "${HF_HOME}" /kaggle/temp
: > "${JAQUA_OUTPUT_DIR}/logs/train.log"

echo "[setup] Installing dependencies and building llama.cpp"
bash setup.sh

NPROC="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
echo "[setup] torchrun processes: ${NPROC}"

quantize_bin() {
  if [[ -x /kaggle/temp/llama.cpp/build/bin/llama-quantize ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/llama-quantize
  elif [[ -x /kaggle/temp/llama.cpp/build/bin/quantize ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/quantize
  else
    echo "Missing llama.cpp quantize binary" >&2
    exit 1
  fi
}

convert_script() {
  if [[ -f /kaggle/temp/llama.cpp/convert_hf_to_gguf.py ]]; then
    echo /kaggle/temp/llama.cpp/convert_hf_to_gguf.py
  elif [[ -f /kaggle/temp/llama.cpp/convert_hf_to_gguf_update.py ]]; then
    echo /kaggle/temp/llama.cpp/convert_hf_to_gguf_update.py
  else
    echo "Missing llama.cpp HF-to-GGUF conversion script" >&2
    exit 1
  fi
}

run_variant() {
  local param_label="$1"
  local variant="$2"
  local base_model="$3"
  local dataset="$4"
  local split="$5"
  local steps="$6"
  local seq_len="$7"
  local micro_batch="$8"
  local grad_accum="$9"
  local lr="${10}"
  local lora_r="${11}"
  local lora_alpha="${12}"
  local max_samples="${13}"

  export JAQUA_PARAM_LABEL="${param_label}"
  export JAQUA_VARIANT="${variant}"
  export JAQUA_BASE_MODEL="${base_model}"
  export JAQUA_DATASET="${dataset}"
  export JAQUA_DATASET_SPLIT="${split}"
  export JAQUA_LORA_STEPS="${steps}"
  export JAQUA_SEQ_LEN="${seq_len}"
  export JAQUA_MICRO_BATCH="${micro_batch}"
  export JAQUA_GRAD_ACCUM="${grad_accum}"
  export JAQUA_LORA_LR="${lr}"
  export JAQUA_LORA_R="${lora_r}"
  export JAQUA_LORA_ALPHA="${lora_alpha}"
  export JAQUA_LORA_DROPOUT="${JAQUA_LORA_DROPOUT:-0.05}"
  export JAQUA_MAX_SAMPLES="${max_samples}"
  export JAQUA_LOG_EVERY="${JAQUA_LOG_EVERY:-20}"
  export JAQUA_SAVE_EVERY="${JAQUA_SAVE_EVERY:-200}"

  local artifact="jaqua-${param_label}-${variant}"
  local merged_dir="${JAQUA_OUTPUT_DIR}/merged/${artifact}-F16"
  local gguf_f16="${JAQUA_OUTPUT_DIR}/gguf/${artifact}-F16.gguf"
  local gguf_q4="${JAQUA_OUTPUT_DIR}/gguf/${artifact}-Q4_K_M.gguf"
  local gguf_q8="${JAQUA_OUTPUT_DIR}/gguf/${artifact}-Q8_0.gguf"

  echo "[train] ${artifact}"
  echo "[train] base=${base_model}"
  echo "[train] data=${dataset} split=${split}"
  torchrun --standalone --nproc_per_node="${NPROC}" lora_train.py 2>&1 | tee -a "${JAQUA_OUTPUT_DIR}/logs/train.log"

  echo "[merge] ${artifact}"
  python lora_merge.py

  echo "[validate] ${artifact}"
  python lora_validate.py

  echo "[gguf] ${artifact} F16"
  python "$(convert_script)" "${merged_dir}" --outfile "${gguf_f16}" --outtype f16

  echo "[gguf] ${artifact} Q4_K_M"
  "$(quantize_bin)" "${gguf_f16}" "${gguf_q4}" Q4_K_M

  echo "[gguf] ${artifact} Q8_0"
  "$(quantize_bin)" "${gguf_f16}" "${gguf_q8}" Q8_0

  if [[ "${JAQUA_KEEP_F16_GGUF:-0}" != "1" ]]; then
    rm -f "${gguf_f16}"
  fi
}

BASE_15_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REASON_27_MODEL="Qwen/Qwen2.5-3B-Instruct"

BASE_DATASET="HuggingFaceH4/ultrachat_200k"
BASE_SPLIT="train_sft"
WEB_DATASET="BitAgent/tool_calling"
WEB_SPLIT="train"
REASON_DATASET="open-r1/OpenR1-Math-220k"
REASON_SPLIT="train"
REASON_WEB_DATASET="open-r1/OpenR1-Math-220k,BitAgent/tool_calling"
REASON_WEB_SPLIT="train,train"

run_variant 1.5b base "${BASE_15_MODEL}" "${BASE_DATASET}" "${BASE_SPLIT}" \
  2400 512 2 4 1e-4 16 32 100000

run_variant 1.5b web "${BASE_15_MODEL}" "${WEB_DATASET}" "${WEB_SPLIT}" \
  1200 512 2 4 8e-5 16 32 80000

run_variant 2.7b reason "${REASON_27_MODEL}" "${REASON_DATASET}" "${REASON_SPLIT}" \
  1800 768 1 8 8e-5 32 64 80000

REASON_MERGED="${JAQUA_OUTPUT_DIR}/merged/jaqua-2.7b-reason-F16"
run_variant 2.7b reason-web "${REASON_MERGED}" "${REASON_WEB_DATASET}" "${REASON_WEB_SPLIT}" \
  1000 768 1 8 5e-5 16 32 80000

echo "[final] Package artifacts"
if [[ "${JAQUA_PACKAGE_MERGED:-0}" == "1" ]]; then
  tar -C "${JAQUA_OUTPUT_DIR}" -czf "${JAQUA_OUTPUT_DIR}/jaqua_artifacts.tar.gz" gguf merged adapters logs
else
  tar -C "${JAQUA_OUTPUT_DIR}" -czf "${JAQUA_OUTPUT_DIR}/jaqua_artifacts.tar.gz" gguf adapters logs
fi

echo "Jaqua LoRA pipeline complete."
echo "GGUF outputs:"
ls -lh "${JAQUA_OUTPUT_DIR}/gguf" | sed -n '1,200p'
