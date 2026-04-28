# Eval

Lightweight harness for sanity-checking model swaps and quant levels. Not a
replacement for full HumanEval+/MBPP+/SWE-bench — just enough to catch
regressions in a few minutes.

## Run

```bash
pip install openai rich
python eval/humaneval_quick.py --model coder-lg --n 20
```

Outputs:

- `eval/runs/<timestamp>/results.jsonl` — per-prompt pass/fail + latency
- `eval/runs/<timestamp>/summary.json`  — pass@1, mean tok/s, p50/p95 latency

## Layout

- `humaneval_quick.py` — 20 hand-picked HumanEval problems, executes generated code
  in a subprocess sandbox with a 5 s timeout
- `prompts/` — drop your own `.md` prompts here for qualitative comparison

## Throughput / latency sweeps (llama-benchy)

For real perf reports — pp/tg at varying context depth, prefix-caching delta,
concurrency saturation — use [eugr/llama-benchy](https://github.com/eugr/llama-benchy)
via the wrapper. It runs in an ephemeral python container attached to the
`kurts-brain_brain` docker network so it can hit `vllm-*:8000` directly and
auto-detect the served model name.

```bash
# default sweep against the running coder-lg-nvfp4 stack (run on the brain host)
bash scripts/bench-llama-benchy.sh coder-lg-nvfp4

# custom sweep
PP="2048 4096" TG="128" DEPTHS="0 4096 16384 32768" CONCURRENCY="1 2 4" \
  bash scripts/bench-llama-benchy.sh coder-lg-nvfp4

# go through LiteLLM instead of vLLM directly
BASE_URL=http://brain-litellm:4000/v1 API_KEY=sk-local MODEL=coder-lg-nvfp4 \
  bash scripts/bench-llama-benchy.sh
```

Results land in `eval/runs/benchy-<stack>-<ts>/{results.md,results.json}`.

`scripts/bench_toks.py` stays as the 5-second smoke test (single stream, TTFT
+ decode tok/s); llama-benchy is the full picture.
