# Claude Code → Local Models

Claude Code (the official Anthropic CLI) calls the Anthropic API. To point it at
local models we use **claude-code-router** — a small proxy that intercepts the
Anthropic protocol and forwards to any OpenAI-compatible backend (our LiteLLM).

Repo: https://github.com/musistudio/claude-code-router

## Install

```bash
npm i -g @musistudio/claude-code-router
```

## Config

Save as `~/.claude-code-router/config.json` on the machine running Claude Code
(your laptop, over OpenVPN):

```json
{
  "PROXY_URL": "",
  "LOG": false,
  "Providers": [
    {
      "name": "kurts-brain",
      "api_base_url": "http://192.168.3.5:4000/v1/chat/completions",
      "api_key": "sk-local",
      "models": ["coder-lg", "coder-md", "agent", "reason", "moe-fast"]
    }
  ],
  "Router": {
    "default":     "kurts-brain,agent",
    "background":  "kurts-brain,reason",
    "think":       "kurts-brain,coder-lg",
    "longContext": "kurts-brain,coder-lg"
  }
}
```

## Run

```bash
ccr code            # wraps `claude` and routes everything through the proxy
```

## Notes

- The router maps Claude's tool-use blocks to OpenAI tool-calls; this works best
  with `agent` (Devstral) which was tuned for Mistral-style tool calling — vLLM
  is started with `--tool-call-parser=mistral --enable-auto-tool-choice` for it.
- For the **Hermes** route (`reason`), tool calls use `--tool-call-parser=hermes`.
- If a model misbehaves with Claude's prompt, switch the `default` route to
  `coder-lg` (more permissive, smarter, slower).
