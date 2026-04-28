#!/usr/bin/env python3
"""Stream a chat completion and report TTFT + decode tok/s."""
import argparse, json, os, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="coder-lg-nvfp4")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--url", default=os.environ.get("LITELLM_URL", "http://localhost:4000/v1/chat/completions"))
ap.add_argument("--key", default=os.environ.get("LITELLM_KEY", "sk-local"))
ap.add_argument("--prompt", default="Write one paragraph about transformer attention. Then count from 1 to 200 comma separated. No other text.")
ap.add_argument("--prompt-file", default=None)
ap.add_argument("--repeat-prompt", type=int, default=1, help="Repeat the prompt N times to grow input length")
ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--no-think", action="store_true", help="Disable Qwen3 thinking mode for raw decode rate")
args = ap.parse_args()

if args.prompt_file:
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        args.prompt = f.read()
prompt = args.prompt * args.repeat_prompt

for run in range(1, args.runs + 1):
    body = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(args.url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {args.key}", "Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None; last_tok_t = t0; first_tok_t = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            if obj.get("usage"): usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0; first_tok_t = now
                    last_tok_t = now
    total = time.perf_counter() - t0
    decode_w = max(last_tok_t - (first_tok_t or t0), 1e-6)
    ct = (usage or {}).get("completion_tokens", 0)
    pt = (usage or {}).get("prompt_tokens", 0)
    print(f"run {run}: prompt={pt}t  completion={ct}t  TTFT={ttft:.3f}s  total={total:.3f}s  decode={ct/decode_w:.2f} tok/s  overall={ct/total:.2f} tok/s", flush=True)
