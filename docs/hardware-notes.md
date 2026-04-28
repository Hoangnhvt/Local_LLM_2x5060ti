# Hardware Notes — 2× RTX 5060 Ti 16 GB on i7-10700K

## Blackwell gotchas

The RTX 50-series (GB20x) reports compute capability **sm_120**. Older CUDA
binaries crash with `no kernel image is available for execution on the device`
or fall back to slow paths.

Minimum versions:

| Component                | Minimum            | Notes |
|--------------------------|--------------------|-------|
| NVIDIA driver            | **R565+** (570 ideal) | `nvidia-smi` should show CUDA 12.8 |
| CUDA toolkit (in image)  | **12.8**           | `nvcr.io/nvidia/pytorch:25.01-py3` or newer |
| PyTorch                  | **2.6 cu128**      | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| vLLM                     | **0.8.4+**         | earlier versions silently miss FlashAttn-3 kernels for sm_120 |
| vLLM (NVFP4)             | **0.19+**          | `compressed-tensors` NVFP4 fast path; needs PyTorch 2.11 + cu130 in image |
| llama.cpp                | recent master      | build with `GGML_CUDA=1 CMAKE_CUDA_ARCHITECTURES=120` |
| bitsandbytes             | **0.45+**          | older wheels lack sm_120 |

## PCIe topology

The 10700K exposes only **16 PCIe 3.0 lanes from the CPU**. With 2 GPUs the board
will typically run them at **x8/x8 PCIe 3.0** (≈ 7.9 GB/s each). For tensor
parallel this means all-reduce traffic between the two cards goes over the chipset.
Expected impact:

- Single-stream inference: ~5-10% slower than NVLink-class boards. Acceptable.
- Batch / high-concurrency: bigger penalty; prefer pipeline-parallel (one model per GPU)
  when serving multiple agents simultaneously.

There is **no NVLink** on 5060 Ti.

## NVFP4 on consumer Blackwell

5th-gen tensor cores (SM 120) execute FP4 matmul natively. NVFP4 (E2M1 + FP8 scales)
beats AWQ INT4 on quality and — in theory — throughput. Caveats on this box:

- vLLM **≥ 0.19** required; pin via `VLLM_IMAGE_NVFP4` in `.env` to avoid bumping AWQ stacks.
- KV @ FP8 is mandatory to hit useful context lengths on 16 GB cards (`--kv-cache-dtype=fp8`).
- TP=2 across PCIe 3.0 x8/x8 with no P2P is the bottleneck; set `NCCL_P2P_DISABLE=1`.
- For Qwen3.6-27B-NVFP4 specifically the `compressed-tensors` path is ~1.6–1.7× slower than
  `modelopt`+MTP siblings on the same GPU. Bench before declaring a winner.
- Quality testing path: run [eval/compare.sh](../eval/compare.sh) to compare `coder-lg`
  (AWQ baseline) vs `coder-lg-nvfp4` on pass@1 + tok/s.

## Power

Each 5060 Ti is 180 W TGP. Two cards = 360 W under full load. Combined with the
10700K (~125 W under coding-burst loads) you want at least an **850 W PSU** with
two separate 8-pin PCIe rails. Cap power if needed:

```bash
# 150 W per card — usually <5% perf loss for inference
sudo nvidia-smi -i 0 -pl 150
sudo nvidia-smi -i 1 -pl 150
```

## Thermals

Stack the cards with at least one slot of air gap. Inference loops sustain >80%
duty cycle, unlike gaming. If the top card hits >82°C, set a custom fan curve via
`nvidia-smi -i 0 --gpu-reset` + `coolbits` or use `nvidia-settings`.

## NUMA / CPU pinning

Single-socket Comet Lake — no NUMA concerns. Don't bother with `numactl`.

## OS

Ubuntu Pro 22.04 or 24.04. Make sure secure boot is **off** or you have signed
NVIDIA modules, otherwise driver install fails silently.

## Verifying the install

```bash
nvidia-smi                                # both GPUs visible, driver ≥ 565
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
docker run --rm --gpus all $VLLM_IMAGE python -c \
  "import torch; print(torch.cuda.get_device_capability(0))"   # → (12, 0)
```
