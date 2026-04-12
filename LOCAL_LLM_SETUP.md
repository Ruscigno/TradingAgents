# Running TradingAgents + OpenClaw with a Local LLM

Both components of the Telegram trading assistant — **TradingAgents** (the analysis pipeline) and **OpenClaw** (the Telegram gateway) — can be pointed at a local LLM served by [Ollama](https://ollama.com), eliminating the dependency on Anthropic's API entirely.

---

## How It Works

### TradingAgents

TradingAgents uses **LangChain** as its LLM abstraction layer (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`) coordinated by a custom factory in [tradingagents/llm_clients/factory.py](tradingagents/llm_clients/factory.py). The factory routes by provider name:

```
"openai" / "ollama" / "openrouter"  →  OpenAIClient (ChatOpenAI)
"anthropic"                          →  AnthropicClient (ChatAnthropic)
"google"                             →  GoogleClient (ChatGoogleGenerativeAI)
```

**Ollama is already a supported provider** — it reuses the `OpenAIClient` since Ollama exposes an OpenAI-compatible REST endpoint. No code changes are required.

### OpenClaw

OpenClaw is configured via `openclaw.config.json5` (local) or `openclaw.config.docker.json5` (Docker). Both files have an LLM/model section that is currently hardcoded to Anthropic. Adding Ollama requires editing those config files only.

---

## Step 1: Install and Start Ollama

```bash
brew install ollama
ollama serve          # starts the local server on http://localhost:11434
```

Pull models that support **tool calling** and have a **large context window** (both are hard requirements for TradingAgents):

```bash
# Deep-thinking model (complex multi-step reasoning)
ollama pull qwen2.5:72b        # recommended — 128K context, full tool use
ollama pull gemma3:27b         # excellent alternative — 128K context (use orieg/gemma3-tools:27b if tool-parsing fails)

# Fast model (analysts, trader, risk nodes)
ollama pull qwen2.5:32b        # balanced performer — ~20GB RAM required
ollama pull gemma3:12b         # lightning fast — fits in 12GB VRAM
```

Verify tool-calling support:
```bash
ollama show qwen2.5:72b | grep -i tools
# should show:  tools: true
```

---

## Step 1.5: Connecting to a Remote Ollama Instance (e.g., Windows PC)

If your local machine (e.g., Mac) cannot run massive models but you have a powerful Windows PC or local server, you can run Ollama entirely remotely:

1. **On your remote machine (Windows PC)**:
   - Add a System Environment Variable named `OLLAMA_HOST` with the value `0.0.0.0` to allow external network connections.
   - Fully restart Ollama from the system tray (quit and reload).
   - Run `ipconfig` (Windows) or `ifconfig` (Linux/Mac) to find the machine's local IP address (e.g., `192.168.1.100`).
2. **On your local machine (Mac)**:
   - In Step 2 & 3 below, replace `http://localhost:11434/v1` or `http://host.docker.internal:11434/v1` with your remote IP address (e.g., `http://192.168.1.100:11434/v1`).

---

## Step 2: Configure TradingAgents

Set the provider and models via the config dictionary (passed to `TradingAgentsGraph`), or pick **Ollama** in the interactive CLI prompt:

```python
config = {
    "llm_provider": "ollama",
    "deep_think_llm":  "qwen2.5:72b",
    "quick_think_llm": "qwen2.5:14b",
    "backend_url": "http://localhost:11434/v1",   # Ollama's OpenAI-compat endpoint
}
```

No API key is needed. The `OpenAIClient` automatically uses `api_key="ollama"` as a placeholder when the provider is `"ollama"`.

> **Note:** The `thinking`/`reasoning_effort` parameters that boost Claude and GPT reasoning are not injected for Ollama — the framework's `_get_provider_kwargs()` only knows Anthropic, OpenAI, and Google. Ollama models receive a plain prompt.

---

## Step 3: Configure OpenClaw

### Option A — Local setup (`openclaw.config.json5`)

Replace the `llm` block:

```json5
// Before (Anthropic):
llm: {
  provider: "anthropic",
  model: "claude-sonnet-4-6",
  apiKey: "${ANTHROPIC_API_KEY}",
},

// After (Ollama):
llm: {
  provider: "ollama",
  model: "qwen2.5:72b",
  baseUrl: "http://localhost:11434/v1",
  apiKey: "ollama",   // placeholder — Ollama requires no real key
},
```

Then restart OpenClaw:
```bash
openclaw gateway restart
```

### Option B — Docker (`openclaw.config.docker.json5`)

1. Update `agents.defaults.model.primary` to reference the Ollama model.
2. Replace the `models.providers.anthropic` block with an `ollama` block.

```json5
"agents": {
  "defaults": {
    "model": {
      "primary": "ollama/qwen2.5:72b",   // was: "anthropic/claude-3-5-sonnet-latest"
    }
  }
},

"models": {
  "providers": {
    "ollama": {
      "baseUrl": "http://host.docker.internal:11434/v1",  // reach host Ollama from inside Docker
      "apiKey": "ollama",
      "models": [
        {
          "id": "qwen2.5:72b",
          "name": "Qwen 2.5 72B",
          "input": ["text"],
          "contextWindow": 131072,
        }
      ],
    }
  }
},
```

Remove `ANTHROPIC_API_KEY` from your `.env` file (or leave it — it will simply be unused).

Then restart the container:
```bash
docker compose restart
```

---

## Model Recommendations

| Role | Model | Memory Needed | Notes |
|---|---|---|---|
| Deep Think / Best | `qwen2.5:72b` | ~40 GB | Extremely smart, best for complex tool chains. |
| Deep Think / Balanced | `gemma3:27b` | ~18 GB | Fast, huge 128K context. Use `orieg/gemma3-tools:27b` if parsing issues arise. |
| Deep Think / Balanced | `qwen2.5:32b` | ~20 GB | Excellent balance, runs partially in VRAM. |
| Quick Think / Fast | `gemma3:12b` | ~8.5 GB | Lightning fast, easily fits entirely in 12GB VRAM GPUs (like RTX 4080 Mobile). |
| Quick Think / Low Spec| `qwen2.5:14b` | ~10 GB | Reliable native tool calling and fast execution. |

For a mixed setup (deep = large, quick = small):
```python
"deep_think_llm":  "qwen2.5:72b",
"quick_think_llm": "qwen2.5:14b",
```

---

## Caveats

| Concern | Detail |
|---|---|
| **Tool use required** | TradingAgents binds tools via `llm.bind_tools()`. Models without tool-calling support will fail at runtime. Always verify with `ollama show <model>`. |
| **Context window** | The agent graph accumulates a large state across multiple hops. Models with < 32K context will truncate intermediate reasoning. Use ≥ 64K where possible. |
| **Reasoning quality** | Local 72B models produce materially weaker financial analysis than Claude Opus 4.6 or GPT-5. Expect lower signal-to-noise in trade decisions. |
| **Thinking tokens** | Claude/OpenAI reasoning-effort parameters are not passed to Ollama. Extended thinking is unavailable locally. |
| **Docker networking** | Inside Docker, use `host.docker.internal:11434` (Mac/Windows) instead of `localhost` to reach the host-side Ollama process. |
| **Hardware** | `qwen2.5:72b` needs ~40 GB VRAM (2× A100, or Mac M2 Ultra/M3 Ultra with 192 GB unified memory). `qwen2.5:14b` fits on a single 16 GB GPU. |
