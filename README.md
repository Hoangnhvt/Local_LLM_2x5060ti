# Kurt's Brain — Local LLM Research Lab

A pragmatic stack for running 9-35B coding/reasoning models on a workstation as the
backend "brain" for local coding agents (OpenCode, Claude Code, Hermes-style assistants).

## Hardware

| Component | Value |
|-----------|-------|
| CPU       | Intel i7-10700K (8c/16t) |
| RAM       | 32 GB DDR4 |
| GPU 0     | NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120) |
| GPU 1     | NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120) |
| Total VRAM | **32 GB** |
| Host      | `kurt@192.168.3.5` (Ubuntu Pro, reachable via OpenVPN) |

> Blackwell consumer GPUs require **CUDA 12.8+**, **PyTorch cu128**, and
> **vLLM ≥ 0.8.4** (or nightly). llama.cpp builds with CUDA 12.8 work fine.
> See [docs/hardware-notes.md](docs/hardware-notes.md).

## Architecture

```
┌─────────────────┐   OpenAI API     ┌──────────────┐    ┌─────────────────────────┐
│  Coding agents  │ ───────────────► │   LiteLLM    │ ─► │  vLLM: qwen-coder-32b   │ (TP=2, both GPUs)
│  OpenCode       │                  │   :4000      │ ─► │  vLLM: devstral-24b     │ (TP=2, both GPUs)
│  Claude Code    │   one base URL   │   gateway    │ ─► │  vLLM: hermes-8b        │ (single GPU)
│  Continue/Cline │                  │              │ ─► │  llama.cpp (overflow)   │
└─────────────────┘                  └──────────────┘    └─────────────────────────┘
```

Only **one** OpenAI-compatible endpoint (`http://192.168.3.5:4000/v1`) is exposed to
agents. LiteLLM handles routing, model aliasing, retries, logging, and budget tracking.
Heavy models (32B/24B) are not co-resident — start one at a time, or use llama.cpp for
lower-cost concurrent loads.

## Recommended Models (curated for 2 × 16 GB)

| Alias              | Model                                          | Format       | Footprint | Use |
|--------------------|------------------------------------------------|--------------|-----------|-----|
| `coder-lg`         | Qwen2.5-Coder-32B-Instruct                     | AWQ 4-bit    | ~20 GB (TP=2) | Primary coder |
| `coder-md`         | Qwen2.5-Coder-14B-Instruct                     | AWQ 4-bit    | ~9 GB     | Fast iteration, single GPU |
| `agent`            | Devstral-Small-2505 (24B, Mistral)             | AWQ 4-bit    | ~14 GB    | Tool-use / agent loops |
| `reason`           | Hermes-3-Llama-3.1-8B                          | FP16 / AWQ   | ~16 / 6 GB| General reasoning, JSON, function calls |
| `moe-fast`         | DeepSeek-Coder-V2-Lite-Instruct (16B MoE)      | AWQ          | ~10 GB    | Fast completions |

See [docs/models.md](docs/models.md) for HF repo IDs, exact tags, and rationale.

## Project Layout

```
.
├── README.md
├── docker-compose.yml          # vLLM services + LiteLLM gateway
├── .env.example
├── stacks/
│   ├── vllm/                   # one compose-overlay per model
│   └── llamacpp/               # GGUF fallback / multi-tenant
├── litellm/
│   └── config.yaml             # model routing, aliases, budgets
├── agents/
│   ├── opencode/               # opencode.json
│   ├── claude-code/            # claude-code-router config
│   └── continue/               # VS Code Continue.dev config
├── scripts/
│   ├── setup-host.sh           # NVIDIA driver, CUDA toolkit, container toolkit, docker
│   ├── pull-models.sh          # HF download into shared volume
│   ├── start.sh STACK          # e.g. `start.sh coder-lg`
│   └── stop.sh
├── eval/
│   ├── README.md
│   ├── humaneval_quick.py      # quick HumanEval-style sanity bench
│   └── prompts/                # in-house coding prompts
├── research/                   # drop-zone for new tech sources (papers, links, PDFs)
│   ├── _inbox/                 # unsorted drops (gitignored)
│   └── <topic>/                # triaged per-topic folders (e.g. nvfp4/)
└── docs/
    ├── hardware-notes.md
    └── models.md
```

## Quick Start (on the remote host)

```bash
# 1. one-time host prep (driver 565+, CUDA 12.8, docker, NVIDIA container toolkit)
sudo bash scripts/setup-host.sh

# 2. pull models you want into ./models (HF cache shared with all containers)
export HF_TOKEN=hf_xxx
bash scripts/pull-models.sh coder-lg agent reason

# 3. start the gateway + a model
bash scripts/start.sh coder-lg

# 4. test
curl http://localhost:4000/v1/models
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local' \
  -d '{"model":"coder-lg","messages":[{"role":"user","content":"write fizzbuzz in rust"}]}'
```

From your laptop (over OpenVPN) point any agent at `http://192.168.3.5:4000/v1`
with API key `sk-local` (override in `.env`).

## Agent Wiring

- **OpenCode** — [agents/opencode/opencode.json](agents/opencode/opencode.json)
- **Claude Code** — [agents/claude-code/README.md](agents/claude-code/README.md) (uses `claude-code-router`)
- **Continue.dev / Cline** — [agents/continue/config.json](agents/continue/config.json)

## Eval

```bash
python eval/humaneval_quick.py --model coder-lg --n 20
```

Results written to `eval/runs/<timestamp>/`. Use to compare quant levels and models.
