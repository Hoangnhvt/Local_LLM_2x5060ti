# Sources — quantization survey for 30B-class on 2× 5060 Ti

## vLLM quantization
- vLLM quant index: https://docs.vllm.ai/en/latest/features/quantization/
- Supported hardware matrix: https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html
- AutoAWQ: https://docs.vllm.ai/en/latest/features/quantization/auto_awq/
- GPTQModel: https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/
- INT4 W4A16: https://docs.vllm.ai/en/latest/features/quantization/int4/
- INT8 W8A8: https://docs.vllm.ai/en/latest/features/quantization/int8/
- FP8 W8A8: https://docs.vllm.ai/en/latest/features/quantization/fp8/
- Quantized KV cache: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- LLM Compressor: https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/
- NVIDIA ModelOpt: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- TorchAO: https://docs.vllm.ai/en/latest/features/quantization/torchao/
- GGUF in vLLM: https://docs.vllm.ai/en/latest/features/quantization/gguf/
- BitsAndBytes: https://docs.vllm.ai/en/latest/features/quantization/bnb/
- MTP spec decode: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/

## Kernels
- Marlin (orig): https://github.com/IST-DASLab/marlin
- Machete (Hopper W4A16): https://github.com/neuralmagic/machete
- CUTLASS NVFP4 GEMM: https://github.com/NVIDIA/cutlass

## Format references
- AWQ paper: https://arxiv.org/abs/2306.00978
- GPTQ paper: https://arxiv.org/abs/2210.17323
- SmoothQuant (W8A8): https://arxiv.org/abs/2211.10438
- NVFP4 / NVIDIA Blackwell FP4: https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/
- OCP Microscaling (MXFP4/MXFP6/MXFP8): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- compressed-tensors (NeuralMagic): https://github.com/neuralmagic/compressed-tensors
- llama.cpp k-quants/i-quants discussion: https://github.com/ggml-org/llama.cpp/discussions/5063
- llama.cpp GGUF quant table: https://github.com/ggerganov/llama.cpp/wiki/Tensor-Encoding-Schemes
- HQQ: https://github.com/mobiusml/hqq
- ExLlamaV2 / EXL2: https://github.com/turboderp/exllamav2
- AQLM: https://github.com/Vahe1994/AQLM
- VPTQ: https://github.com/microsoft/VPTQ
- QuIP#: https://github.com/Cornell-RelaxML/quip-sharp

## HF model search anchors (30B-class)
- Qwen2.5-Coder-32B-Instruct-AWQ: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
- Qwen2.5-Coder-32B-Instruct-GPTQ-Int4: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
- Qwen3.6-27B-NVFP4 (sakamakismile): https://huggingface.co/sakamakismile/Qwen3.6-27B-NVFP4
- Devstral-Small-2505 AWQ: https://huggingface.co/mistralai/Devstral-Small-2505

## Internal cross-refs
- ../nvfp4/NOTES.md — pilot triage, where the engineering work lives
- ../../docs/models.md — current alias catalog
- ../../stacks/vllm/ — overlay configs (coder-lg, coder-lg-nvfp4, etc.)
- ../../litellm/config.yaml — router aliases
