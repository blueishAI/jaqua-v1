#!/usr/bin/env bash
set -euo pipefail

# Kaggle CUDA environment setup for Jaqua LoRA.
export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/kaggle/temp/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

LLAMA_ROOT="/kaggle/temp/llama.cpp"
LLAMA_BIN_DIR="${LLAMA_ROOT}/build/bin"
LLAMA_SRC_DIR="/kaggle/temp/llama.cpp-src"

mkdir -p "${HF_HOME}" "${JAQUA_OUTPUT_DIR:-/kaggle/working/output}" /kaggle/temp "${LLAMA_BIN_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install gguf protobuf
python -m pip uninstall -y tensorflow tensorflow-cpu jax jaxlib torchvision torchaudio || true

apt-get update
apt-get install -y --no-install-recommends wget unzip ca-certificates

download_llama_release() {
  python - <<'PY' > /kaggle/temp/llama_release.env
import json
import urllib.request

with urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest") as resp:
    release = json.load(resp)

assets = release.get("assets", [])
asset = None
for candidate in assets:
    name = candidate.get("name", "")
    if "bin-ubuntu-x64" in name and (name.endswith(".tar.gz") or name.endswith(".zip")):
        asset = candidate
        break

if asset is None:
    names = "\n".join(a.get("name", "") for a in assets)
    raise SystemExit(f"No ubuntu x64 llama.cpp binary asset found. Assets:\n{names}")

print(f"LLAMA_TAG={release['tag_name']}")
print(f"LLAMA_ASSET_NAME={asset['name']}")
print(f"LLAMA_ASSET_URL={asset['browser_download_url']}")
print(f"LLAMA_SOURCE_URL={release['tarball_url']}")
PY

  # shellcheck disable=SC1091
  source /kaggle/temp/llama_release.env

  echo "[setup] downloading llama.cpp ${LLAMA_TAG} binary: ${LLAMA_ASSET_NAME}"
  rm -rf /kaggle/temp/llama-bin-extract "${LLAMA_SRC_DIR}"
  mkdir -p /kaggle/temp/llama-bin-extract "${LLAMA_SRC_DIR}" "${LLAMA_BIN_DIR}"

  wget -q -O "/kaggle/temp/${LLAMA_ASSET_NAME}" "${LLAMA_ASSET_URL}"
  case "${LLAMA_ASSET_NAME}" in
    *.tar.gz)
      tar -xzf "/kaggle/temp/${LLAMA_ASSET_NAME}" -C /kaggle/temp/llama-bin-extract
      ;;
    *.zip)
      unzip -q "/kaggle/temp/${LLAMA_ASSET_NAME}" -d /kaggle/temp/llama-bin-extract
      ;;
    *)
      echo "Unsupported llama.cpp binary archive: ${LLAMA_ASSET_NAME}" >&2
      exit 1
      ;;
  esac

  find /kaggle/temp/llama-bin-extract -type f \( -name 'llama-cli' -o -name 'llama-quantize' -o -name 'quantize' \) -exec cp {} "${LLAMA_BIN_DIR}/" \;
  chmod +x "${LLAMA_BIN_DIR}/"* || true

  if [[ ! -x "${LLAMA_BIN_DIR}/llama-cli" ]] || [[ ! -x "${LLAMA_BIN_DIR}/llama-quantize" && ! -x "${LLAMA_BIN_DIR}/quantize" ]]; then
    echo "Downloaded llama.cpp binary archive did not contain required llama-cli/llama-quantize binaries." >&2
    find /kaggle/temp/llama-bin-extract -maxdepth 3 -type f | sed -n '1,200p' >&2
    exit 1
  fi

  echo "[setup] downloading llama.cpp source for GGUF converter"
  wget -q -O /kaggle/temp/llama-source.tar.gz "${LLAMA_SOURCE_URL}"
  tar -xzf /kaggle/temp/llama-source.tar.gz -C "${LLAMA_SRC_DIR}" --strip-components=1
  ln -sfn "${LLAMA_SRC_DIR}/convert_hf_to_gguf.py" "${LLAMA_ROOT}/convert_hf_to_gguf.py"
}

if [[ ! -x "${LLAMA_BIN_DIR}/llama-cli" ]] || [[ ! -f "${LLAMA_SRC_DIR}/convert_hf_to_gguf.py" ]]; then
  download_llama_release
else
  echo "[setup] using existing llama.cpp binaries in ${LLAMA_BIN_DIR}"
fi

echo "CUDA LoRA setup complete."
