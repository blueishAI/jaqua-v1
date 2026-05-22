#!/usr/bin/env bash
set -euo pipefail

# Kaggle GPU environment setup for Jaqua LoRA.
export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/kaggle/temp/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "${HF_HOME}" /kaggle/working/output /kaggle/temp

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip uninstall -y tensorflow tensorflow-cpu jax jaxlib torchvision torchaudio || true

apt-get update
apt-get install -y --no-install-recommends git cmake build-essential pkg-config

if [[ ! -d /kaggle/temp/llama.cpp ]]; then
  git clone https://github.com/ggerganov/llama.cpp.git /kaggle/temp/llama.cpp
fi

cmake -S /kaggle/temp/llama.cpp -B /kaggle/temp/llama.cpp/build -DGGML_NATIVE=OFF
cmake --build /kaggle/temp/llama.cpp/build -j"$(nproc)"

echo "LoRA setup complete."
