# llama.cpp fallback

When you need to serve **multiple models concurrently from one GPU** (vLLM grabs
one full GPU per model), llama.cpp is the cheaper option. GGUF files are smaller
and you can layer-offload precisely.

## Quick start (Docker)

```bash
docker run -d --gpus all -p 8080:8080 \
  -v /srv/models/gguf:/models \
  --name llamacpp-coder \
  ghcr.io/ggerganov/llama.cpp:server-cuda \
  -m /models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  -c 16384 -ngl 999 --host 0.0.0.0 --port 8080 \
  --parallel 4 --cont-batching
```

Then add to `litellm/config.yaml`:

```yaml
- model_name: coder-md-gguf
  litellm_params:
    model: openai/llamacpp
    api_base: http://host.docker.internal:8080/v1
    api_key: dummy
```

## Recommended GGUF quants (TheBloke / unsloth / bartowski)

| Model                                    | Quant       | VRAM with full offload |
|------------------------------------------|-------------|------------------------|
| Qwen2.5-Coder-32B-Instruct               | Q4_K_M      | ~20 GB → split layers across 2 GPUs with `-ts 1,1` |
| Qwen2.5-Coder-14B-Instruct               | Q5_K_M      | ~11 GB |
| Devstral-Small-2505                      | Q4_K_M      | ~14 GB |
| Hermes-3-Llama-3.1-8B                    | Q6_K        | ~7 GB  |

Build flag for Blackwell if compiling locally:

```bash
cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --config Release -j
```
