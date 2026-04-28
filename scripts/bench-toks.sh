#!/usr/bin/env bash
# Benchmark decode tok/s + TTFT against LiteLLM.
# Usage: bench-toks.sh <model> [max_tokens] [prompt_file_or_-]
set -e
MODEL="${1:-coder-lg-nvfp4}"
MAX_TOK="${2:-512}"
PROMPT_SRC="${3:-}"
KEY="${LITELLM_KEY:-sk-local}"
URL="${LITELLM_URL:-http://localhost:4000/v1/chat/completions}"

if [ -z "$PROMPT_SRC" ]; then
  PROMPT='Write a single-paragraph description of how a transformer attention head works. Then count from 1 to 100, comma-separated, no other text.'
elif [ "$PROMPT_SRC" = "-" ]; then
  PROMPT=$(cat)
else
  PROMPT=$(cat "$PROMPT_SRC")
fi

# JSON-escape the prompt safely
PAYLOAD=$(python3 -c '
import json, sys, os
print(json.dumps({
  "model": os.environ["MODEL"],
  "max_tokens": int(os.environ["MAX_TOK"]),
  "temperature": 0.7,
  "stream": True,
  "stream_options": {"include_usage": True},
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
}))' MODEL="$MODEL" MAX_TOK="$MAX_TOK" PROMPT="$PROMPT")

# Stream and time
python3 - <<'PY'
import json, os, time, urllib.request, sys
url = os.environ["URL"]; key = os.environ["KEY"]; payload = os.environ["PAYLOAD"]
req = urllib.request.Request(url, data=payload.encode(), headers={
  "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.perf_counter(); ttft = None; n_chunks = 0; usage = None; last = t0
with urllib.request.urlopen(req) as r:
  for raw in r:
    line = raw.decode("utf-8", "ignore").strip()
    if not line.startswith("data:"): continue
    data = line[5:].strip()
    if data == "[DONE]": break
    try: obj = json.loads(data)
    except: continue
    if obj.get("usage"): usage = obj["usage"]
    choices = obj.get("choices") or []
    if choices:
      delta = choices[0].get("delta") or {}
      if delta.get("content") or delta.get("reasoning_content"):
        if ttft is None: ttft = time.perf_counter() - t0
        n_chunks += 1
        last = time.perf_counter()
total = time.perf_counter() - t0
decode_t = max(last - (t0 + (ttft or 0)), 1e-6)
ct = (usage or {}).get("completion_tokens", 0)
pt = (usage or {}).get("prompt_tokens", 0)
print(f"prompt_tokens={pt}  completion_tokens={ct}  chunks={n_chunks}")
print(f"TTFT={ttft:.3f}s  total={total:.3f}s  decode_window={decode_t:.3f}s")
if ct: print(f"overall_tok/s = {ct/total:.2f}    decode_tok/s = {ct/decode_t:.2f}")
PY
