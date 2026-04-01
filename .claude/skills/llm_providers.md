---
name: llm_providers
description: Configuration patterns for all supported LLM providers in TradingAgents (OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama).
---

# Skill: LLM Provider Configuration

## Provider Config Reference

Every provider requires `llm_provider`, `deep_think_llm`, and `quick_think_llm`. Provider-specific thinking controls are optional.

```python
# Anthropic (Claude)
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-4-6"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
config["anthropic_effort"] = "high"   # "low" | "medium" | "high" | None

# OpenAI
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5.2"
config["quick_think_llm"] = "gpt-5-mini"
config["openai_reasoning_effort"] = "high"  # "low" | "medium" | "high" | None

# Google (Gemini)
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-2.5-pro"
config["quick_think_llm"] = "gemini-2.0-flash"
config["google_thinking_level"] = "high"  # provider-specific string | None

# xAI (Grok)
config["llm_provider"] = "xai"
config["deep_think_llm"] = "grok-3"
config["quick_think_llm"] = "grok-3-mini"

# OpenRouter (proxy — model IDs include provider prefix)
config["llm_provider"] = "openrouter"
config["deep_think_llm"] = "anthropic/claude-opus-4-6"
config["quick_think_llm"] = "anthropic/claude-haiku-4-5"

# Ollama (local)
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "llama3.3:70b"
config["quick_think_llm"] = "llama3.2:3b"
config["backend_url"] = "http://localhost:11434/v1"
```

## Adding a New Provider

1. Create `tradingagents/llm_clients/my_provider_client.py` extending `BaseLLMClient`.
2. Register it in `tradingagents/llm_clients/__init__.py` inside `create_llm_client()`.
3. Add any provider-specific kwargs extraction to `TradingAgentsGraph._get_provider_kwargs()`.

## Rules

- `deep_think_llm` and `quick_think_llm` must both be valid model IDs for the same `llm_provider`. Mixing providers raises at `create_llm_client()` time.
- Thinking/effort controls are **optional** and silently ignored if the provider client does not support them — check the client implementation before assuming they apply.
- Provider-specific kwargs (e.g., `thinking_level`, `reasoning_effort`) are passed to the LLM constructor, not to individual calls. They apply to every call made by that LLM instance.
