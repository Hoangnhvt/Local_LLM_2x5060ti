# Hermes-style assistants

"Hermes" here means the Nous Research Hermes-3 line, served via the `reason`
alias (Hermes-3-Llama-3.1-8B). It supports:

- ChatML system prompts with strong steerability
- JSON mode (structured output)
- Tool / function calling (`<tool_call>...</tool_call>` format — vLLM's
  `--tool-call-parser=hermes` translates to OpenAI tool-call schema)

## Minimal client example

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.3.5:4000/v1", api_key="sk-local")

resp = client.chat.completions.create(
    model="reason",
    messages=[
        {"role": "system", "content": "You are a precise research assistant."},
        {"role": "user",   "content": "Summarise the differences between MoE and dense LLMs."},
    ],
)
print(resp.choices[0].message.content)
```

## Function calling

```python
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
  }
}]
resp = client.chat.completions.create(model="reason", messages=[...], tools=tools)
```
