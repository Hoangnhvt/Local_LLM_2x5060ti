# Local LLM ROI on 2× RTX 5060 Ti — honest math

> Compiled 2026-04. Numbers are order-of-magnitude estimates for this specific
> build (see `docs/hardware-notes.md`). Re-run with your own electricity rate
> and usage profile before treating as gospel.

Short version: **ROI vs cheap APIs is bad. ROI vs frontier APIs (Claude/GPT-4)
is good if you actually use it daily. ROI vs "privacy + zero per-token anxiety"
is immediate.** Below is the math.

---

## 1. Capex — what the box costs

| Component | Est. price (USD) |
|---|---|
| 2× RTX 5060 Ti 16 GB | ~$900–1,000 |
| CPU + motherboard + 64 GB DDR5 | ~$600–800 |
| 850 W PSU + case + cooling | ~$200–300 |
| 2 TB NVMe (models eat space) | ~$120–180 |
| **Total build** | **~$1,800–2,300** |

Sunk cost. Assume **$2,000** as midpoint.

Optional: depreciate over 3 years → **~$55/month** capex amortization.

---

## 2. Opex — power

Measured/typical for this class of build:

| State | Wattage | Hours/day (typical dev use) |
|---|---|---|
| Idle (model loaded, no traffic) | 100–140 W | 14 h |
| Active inference (TP=2, both GPUs hot) | 380–480 W | 2 h |
| Off / suspend | 5 W | 8 h |

Daily energy ≈ 0.12 × 14 + 0.43 × 2 + 0.005 × 8 ≈ **2.55 kWh/day** ≈ **77 kWh/month**.

At electricity prices:
- VN residential tier (~$0.10/kWh) → **~$7.7/month**
- US average (~$0.16/kWh) → **~$12.3/month**
- EU average (~$0.30/kWh) → **~$23/month**

**Total monthly cost (capex amort + power): ~$63–78/month.**

---

## 3. What you get for it — token throughput

Anchor: `coder-lg` (Qwen2.5-Coder-32B AWQ) at ~30–35 tok/s decode.

| Usage scenario | tok/day | tok/month |
|---|---|---|
| Light (2 h/day, batch 1, ~50 % active gen) | ~110 K | ~3.3 M |
| Medium (4 h/day, batch 2 avg) | ~430 K | ~13 M |
| Heavy / agentic (background loops 8 h/day, batch 4) | ~3.5 M | ~100 M |
| Saturated 24×7 (batch 8) | ~25 M | ~750 M |

(Output tokens only; input tokens are effectively free locally — prefill is much
faster than decode.)

---

## 4. What that would cost on cloud APIs

Pricing snapshot (early 2026, output token rates — input is usually 3–5× cheaper):

| Provider / model | $/M output tok |
|---|---|
| Claude Sonnet 4 / GPT-5-class | ~$15 |
| GPT-4o-mini / Sonnet-mini class | ~$1.5–2.5 |
| DeepSeek V3.x (off-peak) | ~$1.10 |
| Qwen2.5-Coder-32B on Together / OpenRouter | ~$0.20–0.30 |
| Groq / Cerebras (if available) | ~$0.20–0.60 |

### Monthly equivalent cloud spend, by usage scenario

| Usage | vs Qwen-Coder API ($0.25/M) | vs Sonnet-mini ($2/M) | vs Sonnet-4 ($15/M) |
|---|---|---|---|
| Light (3.3 M) | $0.83 | $6.6 | $50 |
| Medium (13 M) | $3.25 | $26 | $195 |
| Heavy (100 M) | $25 | $200 | **$1,500** |
| Saturated (750 M) | $188 | $1,500 | $11,250 |

---

## 5. Break-even tables (months to recoup the $2,000 build)

Local cost = ~$70/month. **Recoup = $2,000 ÷ (cloud_cost − $70).**

| Usage | vs Qwen-API | vs Sonnet-mini | vs Sonnet-4 |
|---|---|---|---|
| Light (3.3 M tok/mo) | **never** (local costs more) | **never** | ~41 mo |
| Medium (13 M) | **never** | **never** (loses $44/mo) | ~16 mo |
| Heavy (100 M) | **never** (loses $45/mo) | ~15 mo | **~1.5 mo** |
| Saturated (750 M) | ~17 mo | ~1.4 mo | **~0.2 mo** |

> "Never" means at that usage level, the cheap API is **cheaper than your
> electricity bill alone**.

---

## 6. Honest verdict

**ROI is OK if and only if at least one of these is true:**

| If you... | ROI verdict |
|---|---|
| Run agentic loops / overnight batch jobs (≥10 M tok/mo) | ✅ Pays back in 1–2 years vs Sonnet-class, instantly vs frontier |
| Need data to never leave the box (NDA, healthcare, IP) | ✅ Cloud equivalent is **HIPAA/private endpoint** tiers — 3–5× pricier |
| Code daily with Continue/Cline/OpenCode and care about latency | ✅ p50 latency beats most APIs, no rate limits, no $ anxiety |
| Want to learn/research quant + serving (this repo's stated purpose) | ✅ The hardware **is** the deliverable |
| Mostly send a few prompts/day to a chat UI | ❌ $5–10/mo of OpenRouter credits beats it |
| Need frontier reasoning quality (Claude Opus, GPT-5) | ❌ 32B local won't match it; cloud is mandatory anyway |

---

## 7. Hidden costs the table doesn't show

- **Your time** — every hour spent debugging vLLM crashes, NCCL P2P, NVFP4
  kernels, and PCIe x8 thrash is a real cost. At $50/h this dwarfs electricity
  within the first month. The repo's `research/nvfp4/NOTES.md` and
  `research/quantization-30b/NOTES.md` show this is non-trivial on Blackwell +
  dual-GPU + no-NVLink.
- **Quality gap** — Qwen2.5-Coder-32B-AWQ ≈ 75–80 % of Sonnet-4 on coding
  benchmarks. For 20 % of tasks you'll still call cloud → hybrid setup, which
  is the realistic optimum.
- **Idle waste** — if the box is on 24×7 but you only use it 2 h/day, ~80 % of
  your power bill is heating the room.
- **Resale floor** — 2× 5060 Ti retains ~50 % value after 2 years → effective
  capex is closer to **$1,000**, halving all break-even months above.

---

## 8. Recommended posture for this box

1. **Keep box suspended when not coding** → cuts power to ~$3–5/mo.
2. **Route via LiteLLM**: cheap/easy prompts → local; hard reasoning →
   Claude/GPT-5 fallback. Hybrid is the only honest answer.
3. **Target ≥30 M tok/mo of real local usage** (agentic loops, eval runs, RAG
   ingest, batch refactors) to make the box pay for itself within ~18 months
   vs Sonnet-class.
4. **Don't compare against $0.25/M open-weight APIs** — at that price, no
   on-prem hardware ever wins. Compare against the *frontier or private-tier*
   alternative you'd actually use.

---

## 9. Bottom line

At medium usage you're paying **~$70/mo for privacy, zero rate limits, low
latency, and a research lab**. Pure $-ROI break-even sits around **12–18 months
only if you hit "heavy" usage vs Sonnet-class APIs**. Below that, treat the
build as a **capability purchase, not an investment**.

---

## Assumptions & caveats

- Capex midpoint $2,000; resale not counted in the break-even tables (would
  roughly halve them).
- Power model assumes the box is suspended 8 h/day and idle the rest. A box
  left on 24×7 at idle adds ~$3–10/mo depending on rate.
- Cloud prices are output-token rates as of early 2026; input is 3–5× cheaper
  but for coding workloads output dominates total $.
- Throughput numbers are single-stream / small-batch; production batch=8+ on
  PCIe x8 without NVLink degrades fast (see
  `research/quantization-30b/NOTES.md` §6).
- "Never break even" entries assume cloud price stays flat. Open-weight API
  prices have only fallen historically — the "never" verdicts get *more*
  pessimistic for local hardware over time, not less.
