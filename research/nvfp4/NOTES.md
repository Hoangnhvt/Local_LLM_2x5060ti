# NVFP4 — triage notes

**Source:** [sakamakismile/Qwen3.6-27B-NVFP4](https://huggingface.co/sakamakismile/Qwen3.6-27B-NVFP4)
(Lna-Lab, NVFP4 W4A4 via `compressed-tensors`)

## What

NVFP4 = NVIDIA's microscaled FP4 (E2M1 weights & activations + FP8 scales). Native
matmul path on Blackwell 5th-gen tensor cores (SM 120). This particular checkpoint
is the **first / "slow" path** in its family — uses `compressed-tensors` format,
preserves the vision tower in BF16, no MTP head.

| | Value |
|---|---|
| Base model | `Qwen/Qwen3.6-27B` (27B params, hybrid Gated DeltaNet + Gated Attention, 16 attn layers) |
| Quant size | **19.7 GB** (single safetensors shard) |
| Context (native) | 262,144 tokens; extensible to 1M |
| License | Apache 2.0 |
| Required vLLM | **≥ 0.19** (was 0.8.5 in our pin — we must bump) |
| Required PyTorch | 2.11 + cu130 in the image |

## Applies to 2× 5060 Ti?

**Yes, but tight and PCIe-bound.** Math:

| Item | Per GPU (TP=2) | Total |
|---|---|---|
| Weights | ~9.85 GB | 19.7 GB |
| Runtime + activations + NCCL | ~1.5 GB | ~3 GB |
| Free for KV (@ 0.92 mem-util) | ~3.4 GB | ~6.8 GB |

KV scaling reference (from model card): 96 GB card freed 64.8 GB for KV → 530K tokens
@ FP8 → ~0.13 MB/token. Our **6.8 GB → ~50K tokens** at FP8 KV. We config **32K** as
the safe default.

Throughput reference: 1× RTX PRO 6000 Blackwell (96 GB) → 58 tok/s decode single-stream,
119 tok/s 2-parallel. Our 2× 5060 Ti will be slower because:

- TP all-reduce per layer rides PCIe 3.0 x8/x8 (no NVLink, no P2P → `NCCL_P2P_DISABLE=1`).
- Memory bandwidth per card (~448 GB/s) is below RTX PRO 6000.

**Realistic estimate: 25–40 tok/s decode single-stream.** Bench will confirm.

## Stack impact (if we adopt)

- Bump `VLLM_IMAGE` from `v0.8.5` to **`v0.19.1`** globally (or per-overlay
  `VLLM_IMAGE_NVFP4` to limit blast radius — that's what we did).
- New overlay [stacks/vllm/coder-lg-nvfp4.yml](../../stacks/vllm/coder-lg-nvfp4.yml) (TP=2, FP8 KV, 32K, max-num-seqs 2).
- New alias `coder-lg-nvfp4` in [litellm/config.yaml](../../litellm/config.yaml).
- New alias in [scripts/pull-models.sh](../../scripts/pull-models.sh).
- New comparison runner [eval/compare.sh](../../eval/compare.sh).

## Maturity

- Format: **prod-ready** on RTX PRO 6000; **pilot** on consumer 5060 Ti (v0.19 is recent;
  TP across PCIe is a known soft spot).
- This specific checkpoint: **superseded** per the author. Faster siblings exist:
  - `Qwen3.6-27B-Text-NVFP4-MTP` — modelopt format + bf16 MTP head + `num_speculative_tokens=3` → 1.67-1.74× throughput.
  - `Carnice-V2-27b-NVFP4-TEXT-MTP`, `Huihui-Qwen3.6-27B-abliterated-NVFP4-(TEXT-)MTP` — same speedup tier.
  Reference: model card "Faster siblings" table.

## Recommendation

- **Pilot now** as `coder-lg-nvfp4` alongside (not replacing) AWQ `coder-lg`.
- **Bench first** with `eval/compare.sh coder-lg coder-lg-nvfp4` to confirm pass@1
  parity and measure tok/s on 2× 5060 Ti.
- **Watch:** modelopt+MTP variants. When one of those drops, add a `coder-lg-nvfp4-mtp`
  overlay with `--speculative_config '{"num_speculative_tokens": 3, "method": "mtp"}'`
  and expect ~1.6× lift.
- **Skip vision** — text-only serving via LiteLLM; vision tower is dead weight here
  (preserved at BF16) but un-removable without re-quantizing.

## Open questions

- [ ] Does vLLM 0.19 break our existing AWQ stacks (Qwen-Coder-32B/14B, Hermes, Devstral)?
      → Plan: keep `VLLM_IMAGE=v0.8.5` for AWQ stacks; only NVFP4 overlay uses `v0.19.1`.
      Revisit when AWQ stacks are re-validated on 0.19.
- [ ] Is the `compressed-tensors` slow path on consumer SM 120 hit by the same kernel
      gap as on the RTX PRO 6000? Bench will tell.
- [ ] Does `--reasoning-parser qwen3` interact correctly with OpenCode/Claude Code router?
