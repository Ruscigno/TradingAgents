# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL RULES — ALWAYS FOLLOW THESE

### 1. LangGraph State — Never Mutate In Place
- **Rule**: LangGraph node functions must **return a new dict** with changed keys; never mutate the incoming `state` dict directly.
- **Why**: LangGraph tracks state transitions by comparing returned dicts. Silent in-place mutations break checkpointing and conditional routing.
- **Implementation**: `return {"market_report": report}` — not `state["market_report"] = report`.

### 2. Trade Date Format
- **Rule**: `trade_date` must always be a `"YYYY-MM-DD"` string throughout the graph.
- **Why**: yfinance and all dataflow functions expect this exact format. A `datetime` object or different format causes silent data fetch failures.
- **Implementation**: Use `datetime.strptime(trade_date, "%Y-%m-%d")` to validate at graph entry; never pass raw user input directly to dataflow functions.

### 3. LLM Provider Consistency
- **Rule**: `deep_think_llm` and `quick_think_llm` must both be models valid for the configured `llm_provider`. Never mix models across providers.
- **Why**: `create_llm_client` routes both models through the same provider client. A model from a different provider will fail at runtime with an opaque API error.
- **Valid combinations**: `"anthropic"` → Claude model IDs; `"openai"` → GPT/o-series IDs; `"google"` → Gemini IDs.

### 4. Analyst Names Must Be Exact
- **Rule**: Entries in `selected_analysts` must be one of: `"market"`, `"social"`, `"news"`, `"fundamentals"`.
- **Why**: `GraphSetup.setup_graph()` uses string matching on these values. An unrecognised name is silently skipped, producing a graph with fewer analysts than intended and no error.

### 5. Per-Agent Memory Isolation
- **Rule**: Each `FinancialSituationMemory` instance must be created separately for each agent; never share a memory object between agents.
- **Why**: Memory stores agent-specific past decisions for reflection. Shared memory corrupts the reflection signal for all agents that share it.

### 6. Tool Call Errors Must Not Crash the Graph
- **Rule**: All dataflow tool functions must catch exceptions per-ticker and return an empty/error string rather than raising.
- **Why**: A single failed yfinance call will abort the entire LangGraph run if uncaught, discarding all previous analyst work.

---

## Commands

```bash
# Install
pip install .
# or with uv (recommended for reproducible builds)
uv sync

# Run CLI
tradingagents
# or directly
python -m cli.main

# Run Python API
python main.py

# Run tests
python -m unittest tests.test_ticker_symbol_handling
# or
python -m pytest

# Quick dataflow smoke test
python test.py
```

## Architecture Overview

TradingAgents is a **multi-agent LLM trading framework** built on **LangGraph**. It simulates a trading firm's research process using specialized agents that collaborate through structured debates.

### Agent Pipeline (sequential phases)

1. **Analyst Phase** — Selected analysts gather data in parallel:
   - Market Analyst: prices, technical indicators (MACD, RSI, Bollinger Bands, ATR)
   - Social Media Analyst: sentiment data
   - News Analyst: news and insider transactions
   - Fundamentals Analyst: balance sheets, cash flows, income statements

2. **Debate Phase** — Bull & Bear Researchers argue investment thesis; Research Manager judges (configurable rounds)

3. **Trading Phase** — Trader creates investment plan

4. **Risk Management Phase** — Aggressive/Neutral/Conservative Analysts debate strategy; (configurable rounds)

5. **Portfolio Phase** — Portfolio Manager makes final buy/hold/sell decision

### Key Layers

| Layer | Location | Purpose |
|-------|----------|---------|
| Graph orchestration | `tradingagents/graph/` | LangGraph workflow, state routing, conditional logic |
| Agents | `tradingagents/agents/` | Analyst, researcher, trader, risk, portfolio agent logic |
| Data | `tradingagents/dataflows/` | Multi-vendor data access (yfinance, Alpha Vantage) |
| LLM clients | `tradingagents/llm_clients/` | Provider-specific clients (OpenAI, Anthropic, Google, xAI) |
| CLI | `cli/` | Rich-based interactive terminal UI |

### Graph Layer Details (`tradingagents/graph/`)

- **`trading_graph.py`**: `TradingAgentsGraph` — top-level entry point; initializes LLMs, memory, builds graph; exposes `propagate(company_name, trade_date)`
- **`setup.py`**: `GraphSetup` — constructs the StateGraph, wires all agent nodes and edges
- **`conditional_logic.py`**: controls graph routing (debate rounds, tool call loops)
- **`propagation.py`**: creates the initial `AgentState` dict for a run
- **`signal_processing.py`**: parses final trading decision from agent output
- **`reflection.py`**: agent learning from past decisions

### State Management

All state flows through TypedDicts defined in `tradingagents/agents/utils/agent_states.py`:
- `AgentState`: top-level state containing all reports, debate states, final decision
- `InvestDebateState`: bull/bear history + research manager decision
- `RiskDebateState`: aggressive/conservative/neutral history + portfolio decision

### LLM Client Pattern

Factory pattern in `tradingagents/llm_clients/`. `TradingAgentsGraph.__init__` instantiates the correct client based on `config["llm_provider"]`. Supports provider-specific features: `openai_reasoning_effort`, `google_thinking_level`, `anthropic_effort`.

### Configuration

`tradingagents/default_config.py` contains `DEFAULT_CONFIG`. Override at runtime:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"       # openai | google | anthropic | xai | openrouter | ollama
config["deep_think_llm"] = "claude-opus-4-6"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
config["max_debate_rounds"] = 2
config["max_risk_discuss_rounds"] = 1
config["selected_analysts"] = ["market", "news", "fundamentals"]  # subset of analysts

ta = TradingAgentsGraph(debug=True, config=config)
final_state, decision = ta.propagate("NVDA", "2026-01-15")
```

Data vendor routing is also in `DEFAULT_CONFIG` under `data_vendors` (per-category) and `tool_vendors` (per-tool overrides). Default is `yfinance` (no API key required).

### Environment / API Keys

Copy `.env.example` to `.env`. Required keys depend on chosen providers (OpenAI, Anthropic, Google, xAI, Alpha Vantage, etc.).
