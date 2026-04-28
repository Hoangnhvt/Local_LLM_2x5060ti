#!/usr/bin/env bash
# Pre-download model weights into $MODELS_DIR so vLLM startup is fast and offline-safe.
#
#   bash scripts/pull-models.sh coder-lg agent reason
#
# Requires HF_TOKEN in env (or .env).
set -euo pipefail

cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . .env && set +a

: "${HF_TOKEN:?set HF_TOKEN in .env}"
MODELS_DIR="${MODELS_DIR:-/srv/models}"
mkdir -p "$MODELS_DIR"

declare -A REPOS=(
  [coder-lg]="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
  [coder-lg-nvfp4]="sakamakismile/Qwen3.6-27B-NVFP4"
  [coder-md]="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
  [agent]="casperhansen/devstral-small-2505-awq"
  [reason]="NousResearch/Hermes-3-Llama-3.1-8B"
  [reason-awq]="solidrust/Hermes-3-Llama-3.1-8B-AWQ"
  [moe-fast]="casperhansen/deepseek-coder-v2-lite-instruct-awq"
)

# Use a dedicated venv to avoid PEP 668 (Ubuntu 24.04 externally-managed env)
HF_VENV="${HF_VENV:-$HOME/.venvs/hf}"
if [[ ! -x "$HF_VENV/bin/hf" ]]; then
  command -v python3 >/dev/null || { echo "python3 missing" >&2; exit 1; }
  python3 -m venv "$HF_VENV"
  "$HF_VENV/bin/pip" install --quiet --upgrade pip
  "$HF_VENV/bin/pip" install --quiet "huggingface_hub" hf_transfer
fi
export PATH="$HF_VENV/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster parallel downloads
export HF_TOKEN   # `hf` reads this directly; no separate login needed

for alias in "$@"; do
  repo="${REPOS[$alias]:-}"
  if [[ -z "$repo" ]]; then
    echo "unknown alias: $alias  (known: ${!REPOS[*]})" >&2
    exit 1
  fi
  echo "==> $alias  ←  $repo"
  hf download "$repo" \
    --local-dir "$MODELS_DIR/$alias" \
    --exclude "*.bin" --exclude "*.pt" --exclude "original/*"
done

echo "DONE. Models in $MODELS_DIR"
ls -lah "$MODELS_DIR"
