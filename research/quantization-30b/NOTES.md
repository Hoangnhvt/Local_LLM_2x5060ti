# Quantization survey for ~30B-class models on 2× RTX 5060 Ti 16 GB

**Status:** triage / desk research, no benchmarks yet.
**Date:** 2025-02 (compiled from vLLM 0.20-dev docs + llama.cpp k/i-quant write-ups + existing repo state).
**Scope:** what quant formats are realistic for serving ~24–32 B dense models on this box, ranked by VRAM fit, expected tok/s, kernel maturity on Blackwell SM 120, and integration cost into the current vLLM + LiteLLM stack.

---

## 1. Hardware envelope (recap)

| Item | Value | Implication |
|---|---|---|
| GPUs | 2× RTX 5060 Ti 16 GB (Blackwell, SM 120) | 32 GB total, **TP=2 mandatory** for ≥20 GB models |
| Mem BW | ~448 GB/s/GPU | Decode is BW-bound; weight bits/param dominates tok/s |
| Interconnect | PCIe 4.0 x8/x8, **no NVLink/P2P** (`NCCL_P2P_DISABLE=1`) | All-reduce on every TP layer hits PCIe; favor lower TP and avoid TP=2 thrash on small batches |
| Compute | SM 120, FP8 + FP4 tensor cores, **no FP6/MXFP6 hw path** | NVFP4 W4A4 is native; INT4 via Marlin emulates on FP16 cores |
| Engine | vLLM 0.8.5 pinned (AWQ stacks); 0.19.1 overlay for NVFP4 | Two parallel images is a real cost — see §6 |
| KV | FP8 E4M3 enabled where supported | ~2× KV headroom vs FP16 |

**Working VRAM budget per GPU (TP=2):**
~16.0 GB total − 1.5 GB CUDA/runtime/activations − ~0.5 GB cuda graphs = **~14 GB usable for weights+KV per GPU**, i.e. **~28 GB total**.

For a 30B dense model that means weights must land **≤ ~22 GB** to leave a sane KV slab (6 GB → ~50–80K tokens @ FP8 KV depending on head/layer config).

---

## 2. Per-format triage

Sorted weight-bits ascending. "GB @ 30B" is total weight footprint for a 30B-param dense model (no embeds extras), TP=2 split evenly.

| Format | Bits/wt | GB @ 30B | KV options | vLLM SM 120 | Maturity | Verdict for 30B here |
|---|---|---|---|---|---|---|
| **NVFP4 W4A4** (compressed-tensors) | 4.0 + FP8 scales (~4.5 eff) | ~17 | FP8 ✓ | ✓ (cutlass + Marlin fallback, vLLM ≥0.19) | Pilot, kernels still maturing | **PILOT** — already wired (`coder-lg-nvfp4`); biggest VRAM headroom + native FP4 cores |
| **MXFP4** (OCP, group=32, E8M0 scales) | ~4.25 | ~16 | FP8 | ✓ Marlin MXFP4 on Blackwell; emulation elsewhere | New (Llama-4 / GPT-OSS use it) | **WATCH** — model availability for 30B coder class is thin; revisit when Qwen ships MXFP4 |
| **AWQ 4-bit** (group=128, Marlin) | 4.25 | ~16 | FP8 | ✓ Marlin (W4A16) on SM 120 | **Mature, default** | **KEEP** — current `coder-lg` baseline. Best perf/quality combo today |
| **GPTQ-Int4** (act-order, g=128) | 4.25 | ~16 | FP8 | ✓ Marlin (W4A16) | Mature | **FALLBACK** — use only if AWQ checkpoint unavailable; quality ~= AWQ |
| **GGUF Q4_K_M** (llama.cpp) | ~4.85 | ~18 | Q4/Q8 KV in llama.cpp | vLLM has GGUF loader (slower kernels) | Mature in llama.cpp; vLLM path is second-class | **SKIP for vLLM**; keep as escape hatch in `stacks/llamacpp/` for offline / single-stream |
| **GGUF IQ4_XS** (i-quant, imatrix) | ~4.25 | ~16 | as above | vLLM ✓ via GGUF | Mature in llama.cpp | **SKIP for vLLM** (kernel slower than Marlin AWQ); good for llama.cpp single-stream where bits matter |
| **EXL2** (~4.0–5.0 bpw, mixed) | configurable | ~15–19 | Q4 cache | ❌ not in vLLM | Mature in ExLlamaV2 | **SKIP** — different engine; doesn't fit current router |
| **FP8 W8A8** (E4M3) | 8.0 | ~30 | FP8 | ✓ (Ada/Hopper/Blackwell native) | Mature | **SKIP @ 30B** — ~30 GB weights barely fits TP=2 with ~0 KV. OK for ≤14 B (`coder-md` candidate). |
| **INT8 W8A8** (SmoothQuant) | 8.0 | ~30 | FP8/INT8 | ✓ | Mature | **SKIP @ 30B** — same VRAM problem as FP8, no perf upside on Blackwell |
| **bitsandbytes NF4** | 4.5 | ~17 | FP16 only | ✓ but slow (no Marlin path) | Mature for fine-tuning, weak for serving | **SKIP** — 2–3× slower decode than AWQ-Marlin |
| **HQQ 4-bit** | 4.0 | ~15 | FP16 | partial (torchao path) | Newer, calibration-free | **WATCH** — interesting because no calibration, but no Marlin-class kernel yet |
| **AQLM / VPTQ / QuIP# 2-bit** | 2.0–2.5 | ~8–9 | FP16 | poor / experimental | Research | **SKIP** — fine-tune required, no good vLLM kernel; 32 B model at 2-bit ≈ 14 B AWQ in quality |
| **W4A8 FP8** (compressed-tensors W4A8-FP8) | 4.0 wt + FP8 act | ~16 | FP8 | ✓ on Blackwell (cutlass) | Newer | **WATCH** — could match AWQ quality with FP8 activations for higher throughput; few 30B coder ckpts published |

> Bits include group-scale overhead (+0.25 bpw for g=128). For ~27 B (Qwen3.6) shave ~10 % off all GB columns; for 32 B (Qwen2.5-Coder) add ~7 %.

---

## 3. VRAM math worked for two anchor models

### 3.1 Qwen2.5-Coder-32B (current `coder-lg`, AWQ)
- Weights: ~32 B × 4.25 bits = ~17 GB → **8.5 GB/GPU** at TP=2.
- Runtime/activations/CUDA graphs: ~1.7 GB/GPU.
- Free for KV per GPU: ~14 − 8.5 − 1.7 = **~3.8 GB → ~32 K tokens FP8 KV** ✓ matches what we configured.
- TP all-reduce: ~64 layers × 2 reduces ≈ tolerable on PCIe x8 at batch 1–2; degrades fast above batch 8.

### 3.2 Qwen3.6-27B (NVFP4 pilot)
- Weights: 27 B × ~4.5 bits eff = ~15 GB → **7.6 GB/GPU**.
- Runtime: ~1.5 GB/GPU.
- Free for KV: ~14 − 7.6 − 1.5 = **~4.9 GB → ~50K FP8 KV per GPU (~50K context window)**; NOTES.md keeps it at 32 K for safety.

### 3.3 Hypothetical FP8 (W8A8) 32B
- Weights: 32 B × 8 = **32 GB → 16 GB/GPU**, leaves **0 GB for KV** → **infeasible**.

### 3.4 Hypothetical 2-bit AQLM 32B
- Weights: ~9 GB → 4.5 GB/GPU, ~9 GB free per GPU → ~120 K KV, but quality drop and zero kernel path in vLLM make it not worth the engineering today.

---

## 4. Throughput expectations (single-stream decode, rough)

Baseline measured anchor: AWQ-Marlin Qwen2.5-Coder-32B, TP=2, ~30–35 tok/s on this box. Other formats scaled by bits/byte and kernel efficiency:

| Format | Expected tok/s (32 B class) | Notes |
|---|---|---|
| AWQ-Marlin W4A16 | **30–35** | reference |
| GPTQ-Int4 Marlin | ~30 | indistinguishable from AWQ in Marlin path |
| NVFP4 W4A4 (cutlass) | **35–45** projected | FP4 tensor cores + lower act bw; PCIe still hurts |
| MXFP4 Marlin | 30–40 | similar to NVFP4 in steady state |
| FP8 W8A8 | n/a (won't fit) | would be ~22–25 if it fit |
| GGUF Q4_K_M (vLLM loader) | 12–18 | non-Marlin path |
| GGUF Q4_K_M (llama.cpp CUDA) | 25–35 single-stream, no batching | use for offline |
| BnB NF4 | 10–14 | dequant overhead |

NVFP4 + MTP speculative decoding (`num_speculative_tokens=3`) projected **1.6–1.7×** on top once vLLM's MTP path stabilizes for compressed-tensors → **~55–70 tok/s** decode ceiling. This is the most interesting unlocked path.

---

## 5. Recommendation matrix vs current aliases

| Alias | Current | Keep / change | Why |
|---|---|---|---|
| `coder-lg` | Qwen2.5-Coder-32B AWQ | **KEEP** as the workhorse | Mature, predictable, fits 32 K ctx |
| `coder-lg-nvfp4` | Qwen3.6-27B NVFP4 | **PROMOTE to A/B** vs `coder-lg` | Bigger headroom, FP4 cores, MTP path |
| `coder-md` | Qwen2.5-Coder-14B AWQ | **CONSIDER FP8 W8A8 14B variant** | 14 GB FP8 fits TP=1 → no all-reduce, expect +20–30 % tok/s |
| `agent` | Devstral-24B AWQ | **KEEP**; pilot W4A8-FP8 when available | Same fit, possibly better acc |
| `moe-fast` | DeepSeek-Coder-V2-Lite AWQ | **KEEP**; watch MXFP4 MoE kernels | MoE quant story is moving fast |
| (new) `coder-lg-mxfp4` | — | **WATCH** | Add when a 27–32 B coder ships in MXFP4 |

### Two concrete next experiments
1. **Bench `coder-lg` (AWQ) vs `coder-lg-nvfp4` head-to-head** with `eval/compare.sh`: HumanEval pass@1, decode tok/s @ batch={1,4,8}, TTFT @ 4 K prompt.
2. **Try W4A8-FP8** on a 24 B (Devstral or Mistral-Small-Instruct) once a compressed-tensors checkpoint is published — compare vs AWQ on the same prompts.

---

## 6. Integration cost / risks

- **vLLM version split** (0.8.5 AWQ vs 0.19.1 NVFP4) is the largest hidden cost. Plan: once NVFP4 stack passes A/B, **migrate all stacks to 0.19.1+** and retire the 0.8.5 image. Validate AWQ Marlin path on 0.19 first (it should be fine — Marlin is upstream stable).
- **PCIe x8/x8 no-NVLink** caps useful TP=2 batch size. Anything above batch ~8 likely regresses vs batch 4. Worth scripting a sweep in `eval/`.
- **KV quant**: FP8 E4M3 KV is fine on SM 120. INT8 KV gives ~no extra benefit and loses quality.
- **LiteLLM router**: alias additions are zero-risk; reasoning-parser interaction (per NVFP4 NOTES) still open for OpenCode.
- **Checkpoint availability**: NVFP4 / MXFP4 / W4A8-FP8 ecosystems for 30 B coder models are thin. AWQ remains the only format with broad first-party coverage.

---

## 7. Open questions to resolve before promoting NVFP4 to default

1. NVFP4 cutlass kernel quality on SM 120 at small batches — is it actually faster than Marlin-AWQ at batch=1?
2. MTP speculative decoding stability with compressed-tensors weights in vLLM 0.19+ (still flagged experimental in some PRs).
3. Quality delta on HumanEval+ / SWE-bench Lite between Qwen3.6-27B-NVFP4 and Qwen2.5-Coder-32B-AWQ — is the 5B fewer params offset by the newer base?
4. Whether to retire `coder-md` (AWQ 14 B) in favor of an FP8 14 B running TP=1 on a single card, freeing the second GPU for `reason` or `agent` co-tenancy.

---

## 8. What to skip and why (one-liners)

- **bitsandbytes NF4** — no Marlin-class kernel → decode-bound on dequant.
- **EXL2** — not in vLLM; would fork the stack.
- **AQLM / VPTQ / QuIP#** — research-grade, requires fine-tuning, no good kernel.
- **FP8/INT8 W8A8 at 30 B** — won't fit TP=2 with usable KV.
- **GGUF inside vLLM** — kernel path is slower than Marlin; use llama.cpp directly for the GGUF use case instead.
