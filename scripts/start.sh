#!/usr/bin/env bash
# Bring up the LiteLLM gateway plus one or more vLLM model services.
#
#   bash scripts/start.sh coder-lg                # heavy single-model
#   bash scripts/start.sh agent reason            # two co-resident models
#
# Each alias maps to a compose overlay in stacks/vllm/<alias>.yml
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then
  echo "usage: $0 ALIAS [ALIAS ...]   (see stacks/vllm/*.yml)"
  exit 1
fi

OVERLAYS=(-f docker-compose.yml)
for a in "$@"; do
  f="stacks/vllm/${a}.yml"
  [[ -f "$f" ]] || { echo "no overlay for $a — expected $f"; exit 1; }
  OVERLAYS+=(-f "$f")
done

docker compose "${OVERLAYS[@]}" up -d
docker compose "${OVERLAYS[@]}" ps
echo
echo "Gateway:  http://$(hostname -I | awk '{print $1}'):${LITELLM_PORT:-4000}/v1"
