#!/usr/bin/env bash
# Wrapper around eugr/llama-benchy:
#   https://github.com/eugr/llama-benchy
#
# llama-bench-style measurements (pp/tg @ varying depth, prefix caching,
# concurrency) against any OpenAI-compatible endpoint. Far more thorough
# than scripts/bench_toks.py — use this for real perf reports, keep
# bench_toks.py for quick smoke tests.
#
# By default this runs llama-benchy inside an ephemeral python:3.12-slim
# container attached to the kurts-brain_brain docker network so it can
# resolve service hostnames like `vllm-coder-lg-nvfp4`. That avoids
# LiteLLM's prefix-caching / streaming option rewrites and lets
# llama-benchy auto-detect the model name from /v1/models.
#
# Usage (run on the brain host):
#   bash scripts/bench-llama-benchy.sh                       # coder-lg-nvfp4, default sweep
#   bash scripts/bench-llama-benchy.sh coder-lg              # different stack
#   STACK=coder-lg-nvfp4 PP="2048 4096" DEPTHS="0 4096 16384" \
#     bash scripts/bench-llama-benchy.sh
#
# To benchmark via LiteLLM (in-network) instead of vLLM directly:
#   BASE_URL=http://brain-litellm:4000/v1 API_KEY=sk-local MODEL=coder-lg-nvfp4 \
#     bash scripts/bench-llama-benchy.sh
#
# Env:
#   STACK         vLLM compose service short name (default: coder-lg-nvfp4)
#   BASE_URL      full /v1 URL (overrides STACK-derived default)
#   MODEL         served model name (auto-detect from /v1/models when blank)
#   API_KEY       bearer token (default EMPTY)
#   PP            space-separated prompt-token sweep (default: 2048)
#   TG            space-separated generation sweep (default: 128)
#   DEPTHS        space-separated context depths    (default: 0 4096 16384)
#   CONCURRENCY   space-separated client counts     (default: 1)
#   RUNS          iterations per cell               (default: 3)
#   PREFIX_CACHE  1 to add --enable-prefix-caching  (default: 1)
#   EXTRA         any extra llama-benchy flags appended verbatim
#   NETWORK       docker network to attach to       (default: kurts-brain_brain)
#   IMAGE         python image                      (default: python:3.12-slim)
#   OUT_DIR       results dir on host               (default: eval/runs/benchy-<stack>-<ts>)
set -euo pipefail
cd "$(dirname "$0")/.."

STACK="${STACK:-${1:-coder-lg-nvfp4}}"
BASE_URL="${BASE_URL:-http://vllm-${STACK}:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
PP="${PP:-2048}"
TG="${TG:-128}"
DEPTHS="${DEPTHS:-0 4096 16384}"
CONCURRENCY="${CONCURRENCY:-1}"
RUNS="${RUNS:-3}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
EXTRA="${EXTRA:-}"
NETWORK="${NETWORK:-kurts-brain_brain}"
IMAGE="${IMAGE:-python:3.12-slim}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-eval/runs/benchy-${STACK}-${TS}}"
mkdir -p "$OUT_DIR"
ABS_OUT="$(cd "$OUT_DIR" && pwd)"

echo "endpoint : $BASE_URL"
echo "model    : ${MODEL:-<auto>}"
echo "pp       : $PP"
echo "tg       : $TG"
echo "depths   : $DEPTHS"
echo "conc.    : $CONCURRENCY"
echo "runs     : $RUNS"
echo "out      : $OUT_DIR"
echo "network  : $NETWORK"
echo

ARGS=(
  --base-url "$BASE_URL"
  --api-key  "$API_KEY"
  --pp $PP
  --tg $TG
  --depth $DEPTHS
  --concurrency $CONCURRENCY
  --runs "$RUNS"
  --latency-mode generation
  --save-result /out/results.json
  --format json
)
[[ -n "${MODEL:-}" ]]       && ARGS+=( --model "$MODEL" )
[[ "$PREFIX_CACHE" = "1" ]] && ARGS+=( --enable-prefix-caching )
# shellcheck disable=SC2206
[[ -n "$EXTRA" ]]           && ARGS+=( $EXTRA )

# Quote args safely for the in-container bash -c.
QUOTED_ARGS=$(printf ' %q' "${ARGS[@]}")

# Persist HF tokenizer downloads across runs.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

docker run --rm \
  --network "$NETWORK" \
  -v "$ABS_OUT":/out \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  bash -c "pip install --quiet --root-user-action=ignore llama-benchy && llama-benchy${QUOTED_ARGS}" \
  2>&1 | tee "$OUT_DIR/results.md"

echo
echo "→ $OUT_DIR"
