#!/usr/bin/env bash
# Run eval/humaneval_quick.py against multiple models and print a side-by-side summary.
#
#   bash eval/compare.sh coder-lg coder-lg-nvfp4
#
# Each model must already be running (or routable via LiteLLM).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL_A MODEL_B [MODEL_C ...]"; exit 1
fi

N="${N:-20}"
TS=$(date +%Y%m%d-%H%M%S)
OUT="eval/runs/compare-$TS"
mkdir -p "$OUT"

for m in "$@"; do
  echo "=========================================="
  echo "  $m"
  echo "=========================================="
  python eval/humaneval_quick.py --model "$m" --n "$N"
  # find the latest run dir for this invocation
  latest=$(ls -1dt eval/runs/2*/ 2>/dev/null | head -1)
  cp "$latest/summary.json" "$OUT/$m.json"
  cp "$latest/results.jsonl" "$OUT/$m.jsonl"
done

echo
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
printf "%-22s %-8s %-10s %-12s %-12s %-14s\n" model n pass@1 p50_lat p95_lat tok/s
for m in "$@"; do
  python - "$OUT/$m.json" "$m" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:<22} {d['n']:<8} {d['pass@1']:<10} {d['p50_latency_s']:<12} {d['p95_latency_s']:<12} {d['mean_tok_per_s']:<14}")
PY
done | tee "$OUT/summary.txt"

echo
echo "→ $OUT"
