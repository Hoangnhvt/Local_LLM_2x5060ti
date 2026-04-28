# Model Catalog

Curated for **2 × 16 GB** Blackwell. All models below have been chosen because
they are (a) strong at coding/agent tasks, (b) actively maintained, and (c) have
quantized weights that fit cleanly on this hardware.

## Primary coder — `coder-lg`

- **Repo:** `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- **Quant:** AWQ 4-bit (group 128)
- **VRAM:** ~20 GB at 32K context with TP=2
- **vLLM args:** `--tensor-parallel-size 2 --quantization awq_marlin --max-model-len 32768`
- **Why:** state-of-the-art open coder in this class; near-Sonnet on HumanEval+/MBPP+.
- **Alt:** `Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4` if AWQ fails to compile on your vLLM build.

## NVFP4 pilot — `coder-lg-nvfp4`

- **Repo:** `sakamakismile/Qwen3.6-27B-NVFP4`
- **Quant:** NVFP4 (W4A4 + FP8 scales) via `compressed-tensors`
- **VRAM:** ~19.7 GB weights + KV @ FP8 — fits at 32K context with TP=2 on 2×16 GB
- **vLLM image:** **≥ 0.19** (override `VLLM_IMAGE_NVFP4` in `.env`)
- **Why:** native FP4 matmul on Blackwell SM 120 — not just FP4 storage. Pilot vs `coder-lg` (AWQ).
- **Caveats:** PCIe 3.0 x8/x8 + no NVLink → expect 25–40 tok/s decode (vs 58 on RTX PRO 6000).
  This checkpoint is the slow `compressed-tensors` path; modelopt+MTP siblings are 1.6–1.7× faster
  on the same hardware — watch for those landing.
- **Triage:** [../research/nvfp4/NOTES.md](../research/nvfp4/NOTES.md)

## Mid coder — `coder-md`

- **Repo:** `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- **VRAM:** ~9 GB single GPU
- **Use:** fast inner-loop completion, FIM, repo-RAG. Frees the second GPU for another model.

## Agent / tool-use — `agent`

- **Repo:** `mistralai/Devstral-Small-2505` (24B, Apache 2.0)
- **Quant:** quantize yourself to AWQ, or use community `casperhansen/devstral-small-2505-awq`
- **VRAM:** ~14 GB AWQ at 32K
- **Why:** Mistral fine-tuned specifically for SWE-bench-style agent loops; strong tool calling,
  long-horizon edits, and repo navigation. Pairs naturally with OpenCode/Claude Code workflows.

## Reasoning / chat — `reason`

- **Repo:** `NousResearch/Hermes-3-Llama-3.1-8B`
- **Quant:** FP16 (~16 GB) or AWQ (~6 GB) via `solidrust/Hermes-3-Llama-3.1-8B-AWQ`
- **Why:** classic Hermes — excellent system-prompt steering, JSON mode, function calling,
  uncensored research use. Lightweight enough to keep resident alongside `agent`.

## Fast MoE — `moe-fast`

- **Repo:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (16B / 2.4B active)
- **Quant:** AWQ via `casperhansen/deepseek-coder-v2-lite-instruct-awq`
- **VRAM:** ~10 GB
- **Why:** MoE → only 2.4B active params per token → very high TPS. Great for completion plugins.

## Resident loadouts

Pick one row at a time:

| Loadout            | GPU 0           | GPU 1           | Total VRAM | Concurrency |
|--------------------|-----------------|-----------------|------------|-------------|
| Heavy coder (AWQ)  | coder-lg (TP)   | coder-lg (TP)   | ~20 GB     | 1 model     |
| Heavy coder (FP4)  | coder-lg-nvfp4 (TP) | coder-lg-nvfp4 (TP) | ~20 GB | 1 model |
| Agent + reason     | agent           | reason          | ~20 GB     | 2 models    |
| Mid coder + reason | coder-md        | reason          | ~15 GB     | 2 models    |
| Fast everything    | moe-fast        | reason (AWQ)    | ~16 GB     | 2 models    |

LiteLLM aliases stay the same regardless of which loadout is hot — agents
don't need reconfiguring when you swap.
