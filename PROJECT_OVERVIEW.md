# TradingAgents — Project Overview

> **Version**: 0.2.2 | **License**: Apache 2.0 | **Paper**: [arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138)

---

## 1. Goals & Purpose

TradingAgents is a **research-grade, multi-agent LLM framework** that simulates the collaborative decision-making process of a real-world trading firm. Rather than asking a single LLM "should I buy NVDA?", the system decomposes the problem into specialised analyst roles that gather data independently, debate competing interpretations, and converge on a structured recommendation through a layered review process.

**Primary goals:**

- Demonstrate that structured multi-agent debate improves the quality and robustness of LLM-generated trading signals compared to single-agent approaches (the core thesis of the companion research paper).
- Provide a configurable, provider-agnostic framework that researchers and developers can extend to test new agent configurations, data sources, and LLM models.
- Ship a production-quality CLI that non-engineers can use to run analyses against real market data.

**What it is not:** A live trading system, a backtesting engine, or a financial advisory product. All outputs are for research purposes.

---

## 2. Repository Layout

```
TradingAgents/
├── tradingagents/               # Core framework package
│   ├── graph/                   # LangGraph orchestration layer
│   ├── agents/                  # All agent implementations
│   │   ├── analysts/            # Data-gathering agents
│   │   ├── researchers/         # Investment debate team
│   │   ├── managers/            # Research & portfolio managers
│   │   ├── trader/              # Trade plan generator
│   │   ├── risk_mgmt/           # Risk debate team
│   │   └── utils/               # State, memory, tool wrappers
│   ├── dataflows/               # Multi-vendor data integration
│   ├── llm_clients/             # Provider-specific LLM clients
│   └── default_config.py        # Central configuration defaults
├── cli/                         # Rich interactive terminal UI
├── tests/                       # Unit test suite
├── main.py                      # Minimal Python API example
├── test.py                      # Dataflow smoke test
├── CLAUDE.md                    # Development rules for Claude Code
└── .claude/                     # Claude Code agents & skills
```

---

## 3. Architecture

### 3.1 Five-Phase Trading Pipeline

Every call to `TradingAgentsGraph.propagate(ticker, date)` passes through five sequential phases:

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Analyst Team (parallel tool-calling loops)        │
│  Market · Social Media · News · Fundamentals                │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Investment Debate                                 │
│  Bull Researcher ↔ Bear Researcher  →  Research Manager     │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Trading                                           │
│  Trader (synthesises investment plan)                       │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Risk Debate                                       │
│  Aggressive ↔ Conservative ↔ Neutral  →  Portfolio Manager  │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Signal Extraction                                 │
│  BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL               │
└─────────────────────────────────────────────────────────────┘
```

Each phase maps onto a distinct section of the LangGraph `StateGraph`. The number of debate rounds in phases 2 and 4 is configurable.

### 3.2 Layer Map

| Layer | Path | Responsibility |
|---|---|---|
| **Graph orchestration** | `tradingagents/graph/` | Builds and runs the LangGraph StateGraph; state initialization; signal extraction; reflection |
| **Agent logic** | `tradingagents/agents/` | System prompts and tool usage for each agent role |
| **Data tools** | `tradingagents/agents/utils/*_tools.py` | LangChain `@tool` wrappers; abstract interface agents call |
| **Data vendor routing** | `tradingagents/dataflows/interface.py` | Maps tool names → vendor implementations; handles fallback on rate limits |
| **Vendor implementations** | `tradingagents/dataflows/y_finance.py`, `alpha_vantage*.py` | Raw data retrieval and formatting |
| **LLM clients** | `tradingagents/llm_clients/` | Provider-specific client creation; normalises API differences |
| **Memory** | `tradingagents/agents/utils/memory.py` | BM25-based per-agent episodic memory |
| **CLI** | `cli/` | Rich terminal UI, user configuration, live progress display |

---

## 4. Component Deep-Dives

### 4.1 Graph Orchestration (`tradingagents/graph/`)

**`TradingAgentsGraph`** (`trading_graph.py`) is the single public entry point. On construction it:

1. Instantiates two LLM objects from the configured provider — `quick_thinking_llm` (analysts, researchers, trader, risk agents) and `deep_thinking_llm` (Research Manager, Portfolio Manager).
2. Creates five independent `FinancialSituationMemory` instances — one per memory-aware agent (Bull, Bear, Trader, Research Manager, Portfolio Manager).
3. Builds `ToolNode` objects mapping each analyst type to its allowed tools.
4. Delegates graph construction to `GraphSetup`.

**`GraphSetup`** (`setup.py`) constructs the `StateGraph`:

- Analyst nodes are added conditionally based on `selected_analysts`. Each analyst gets three nodes: the analyst itself, a `ToolNode` for executing tool calls, and a message-clear node to flush messages before handing off to the next analyst (required for Anthropic compatibility).
- Analysts are wired in series in the order they appear in `selected_analysts`. The last analyst connects to Bull Researcher.
- All downstream nodes (researchers, trader, risk team, portfolio manager) are always present regardless of analyst selection.

**`ConditionalLogic`** (`conditional_logic.py`) implements all routing functions:

- For each analyst: routes back to the tool node if the LLM produced a tool call, otherwise forwards to the message-clear node.
- For the investment debate: increments round counter; routes between Bull and Bear up to `max_debate_rounds`, then exits to Research Manager.
- For the risk debate: routes Aggressive → Conservative → Neutral → Aggressive in a cycle up to `max_risk_discuss_rounds`, then exits to Portfolio Manager.

**`Reflector`** (`reflection.py`) supports post-trade learning. After calling `reflect_and_remember(returns_losses)`, it generates a natural-language reflection on each agent's past behaviour and stores it in that agent's `FinancialSituationMemory`. The memory is then retrieved via BM25 at the start of the next analysis.

**`SignalProcessor`** (`signal_processing.py`) parses the Portfolio Manager's free-text output to extract one of five canonical signals: `BUY`, `OVERWEIGHT`, `HOLD`, `UNDERWEIGHT`, `SELL`.

### 4.2 Agent Implementations (`tradingagents/agents/`)

All agents follow the same **factory function pattern**:

```python
def create_[agent_name](llm, [optional_memory]) -> Callable[[AgentState], dict]:
    def [agent_name]_node(state: AgentState) -> dict:
        # 1. Extract relevant state fields
        # 2. Build prompt (system + human messages), optionally with memory retrieval
        # 3. Call llm.invoke(messages) or use tool-binding
        # 4. Return dict with ONLY the keys this node updates
    return [agent_name]_node
```

The factory pattern allows partial application of LLM and memory objects at graph-build time, keeping node functions pure state-in / state-out.

#### Analysts

| Agent | Uses `deep_thinking_llm`? | Tool calls | Output key |
|---|---|---|---|
| Market Analyst | No | `get_stock_data`, `get_indicators` | `market_report` |
| Social Media Analyst | No | `get_news` | `sentiment_report` |
| News Analyst | No | `get_news`, `get_global_news`, `get_insider_transactions` | `news_report` |
| Fundamentals Analyst | No | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | `fundamentals_report` |

Analysts run in tool-calling loops: the LLM decides which tools to call, the `ToolNode` executes them, and results are appended to `messages` until the LLM produces a final report with no further tool calls.

The Market Analyst is prompted to select up to 8 complementary technical indicators from a catalogue of ~30 (MACD, RSI, Bollinger Bands, ATR, VWMA, SMA/EMA variants, MFI, etc.) based on what it deems relevant for the current market conditions.

#### Researchers

**Bull Researcher** and **Bear Researcher** alternate in debate. Each reads the full set of analyst reports from state plus any relevant memories retrieved by BM25, then constructs an argument either advocating for or against the investment. They see each other's prior arguments via `investment_debate_state.history`.

**Research Manager** acts as judge. It reads the complete debate history and analyst reports, adjudicates, and produces:
- A definitive `judge_decision` stored in `investment_debate_state`
- A detailed `investment_plan` returned as a top-level state key

This uses `deep_thinking_llm`.

#### Trader

The Trader receives the `investment_plan` from the Research Manager along with its own memories from past trades. It translates the strategic plan into a concrete `trader_investment_plan` specifying position direction, rationale, and execution considerations.

#### Risk Management Team

Three agents debate in a rotating cycle:

- **Aggressive Analyst** — advocates for maximum position sizing, emphasises upside
- **Conservative Analyst** — prioritises capital preservation, questions aggressive assumptions
- **Neutral Analyst** — seeks a balanced position, challenges both extremes

All three see the `trader_investment_plan` and the accumulating `risk_debate_state.history`.

**Portfolio Manager** adjudicates using `deep_thinking_llm` and outputs a structured `final_trade_decision` using the **five-tier rating scale**: `Buy`, `Overweight`, `Hold`, `Underweight`, `Sell`. The output includes entry strategy, position sizing guidance, risk levels, and an investment thesis anchored in the debate evidence.

### 4.3 State Management (`tradingagents/agents/utils/agent_states.py`)

The entire pipeline state is a single `AgentState` TypedDict that flows through LangGraph:

```python
AgentState
├── messages: list                     # LangChain tool-call message history (cleared between analysts)
├── company_of_interest: str           # Ticker symbol (e.g., "NVDA", "7203.T")
├── trade_date: str                    # "YYYY-MM-DD"
├── sender: str                        # Name of last agent to write
├── market_report: str
├── sentiment_report: str
├── news_report: str
├── fundamentals_report: str
├── investment_debate_state: InvestDebateState
│   ├── bull_history / bear_history / history: str  # Accumulated debate text
│   ├── current_response: str
│   ├── judge_decision: str
│   └── count: int
├── investment_plan: str               # Research Manager output
├── trader_investment_plan: str
├── risk_debate_state: RiskDebateState
│   ├── aggressive_history / conservative_history / neutral_history / history: str
│   ├── latest_speaker: str
│   ├── current_*_response: str        # Per-analyst latest turn
│   ├── judge_decision: str
│   └── count: int
└── final_trade_decision: str          # Portfolio Manager output
```

**Key design constraint**: LangGraph tracks state transitions by comparing the dict returned by each node to the previous state. Nodes must return a dict containing only the keys they modified — mutating the input dict directly is silently ignored.

### 4.4 Memory System (`tradingagents/agents/utils/memory.py`)

`FinancialSituationMemory` implements BM25 (Best Matching 25) lexical similarity retrieval. Design choices:

- **No embeddings / no external API**: BM25 is entirely offline and works with any LLM provider, including Ollama. This avoids embedding cost and latency on every analysis.
- **Per-agent isolation**: Five separate instances are created in `TradingAgentsGraph.__init__`. Each stores the reflection history of a single agent role.
- **Interface**: `add_situations([(situation_text, advice_text), ...])` stores memories; `get_memories(current_situation, n=3)` returns the top-n most similar past situations as formatted text injected into the agent's system prompt.
- **Persistence**: Memory is in-process only. It survives across multiple `propagate()` calls on the same `TradingAgentsGraph` instance but is lost when the process exits. There is no built-in persistence to disk or Redis (unlike the market-data-service companion project).

### 4.5 Data Layer (`tradingagents/dataflows/`)

#### Vendor Routing

`interface.py` provides `route_to_vendor(tool_name, *args)` which:

1. Looks up the configured vendor for the tool (tool-level override → category-level config → hardcoded default).
2. Calls the corresponding vendor implementation function.
3. On `AlphaVantageRateLimitError`, automatically falls back to the next available vendor.

This means agents never need to know which vendor is serving data — they call abstract tool functions and the routing layer handles the rest.

#### Tool → Vendor Mapping

| Abstract Tool | yfinance impl | Alpha Vantage impl |
|---|---|---|
| `get_stock_data` | `get_YFin_data_online()` | AV stock module |
| `get_indicators` | `get_stock_stats_indicators_window()` | AV indicator module |
| `get_fundamentals` | yfinance `.info` dict | AV fundamentals module |
| `get_balance_sheet` | yfinance `.balance_sheet` | AV fundamentals module |
| `get_cashflow` | yfinance `.cashflow` | AV fundamentals module |
| `get_income_statement` | yfinance `.financials` | AV fundamentals module |
| `get_news` | yfinance `.news` | AV news module |
| `get_global_news` | Placeholder | AV news module |
| `get_insider_transactions` | yfinance `.insider_transactions` | — |

#### Technical Indicators

`stockstats_utils.py` wraps the `stockstats` library to compute ~30 indicators from OHLCV data: MACD (`macd`, `macds`, `macdh`), RSI, Bollinger Bands (`boll`, `boll_ub`, `boll_lb`), ATR, VWMA, SMA/EMA variants (10/50/200-period), and MFI.

#### Data Caching

Raw yfinance downloads are cached to `tradingagents/dataflows/data_cache/` as CSV files. This avoids re-fetching the same 15-year history on repeated runs. The cache is keyed by ticker and date range.

### 4.6 LLM Client Layer (`tradingagents/llm_clients/`)

A factory function `create_llm_client(provider, model, base_url, **kwargs)` instantiates the appropriate client. All clients extend `BaseLLMClient` and expose a single `get_llm()` method returning a LangChain-compatible LLM object.

Provider-specific thinking/reasoning controls are passed as constructor kwargs:

| Provider | Config key | Values |
|---|---|---|
| `openai` | `openai_reasoning_effort` | `"low"` / `"medium"` / `"high"` |
| `google` | `google_thinking_level` | `"high"` / `"minimal"` |
| `anthropic` | `anthropic_effort` | `"low"` / `"medium"` / `"high"` |

`xai`, `openrouter`, and `ollama` all route through the OpenAI client using a custom `base_url`.

### 4.7 CLI (`cli/`)

The CLI is a full-featured terminal application built with [Rich](https://github.com/Textualize/rich) and [Questionary](https://github.com/tmbo/questionary). It runs `TradingAgentsGraph` in a background thread while the main thread drives a live display.

**User configuration flow** (7 steps):

1. Ticker symbol (exchange-qualified tickers like `7203.T` supported)
2. Analysis date (`YYYY-MM-DD`)
3. Analyst team selection (any subset of the four analyst types)
4. Research depth (maps to `max_debate_rounds` and `max_risk_discuss_rounds`)
5. LLM provider
6. Shallow and deep model selection for the chosen provider
7. Provider-specific options (reasoning effort, thinking level, etc.)

**Live display** shows a multi-pane layout: agent status table, streaming messages, completed analyst reports, running statistics (LLM calls, tool calls, token counts, elapsed time).

Results are saved as JSON to `eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json`, containing the complete state including all analyst reports, full debate transcripts, and the final decision.

---

## 5. Configuration Reference

All configuration lives in `tradingagents/default_config.py` as `DEFAULT_CONFIG`. Always copy before modifying:

```python
from tradingagents.default_config import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
```

| Key | Default | Description |
|---|---|---|
| `llm_provider` | `"openai"` | `openai` / `google` / `anthropic` / `xai` / `openrouter` / `ollama` |
| `deep_think_llm` | `"gpt-5.2"` | Model ID for Research Manager and Portfolio Manager |
| `quick_think_llm` | `"gpt-5-mini"` | Model ID for all other agents |
| `backend_url` | OpenAI default | Override for OpenRouter, Ollama, etc. |
| `max_debate_rounds` | `1` | Investment debate iterations (Bull ↔ Bear cycles) |
| `max_risk_discuss_rounds` | `1` | Risk debate iterations |
| `max_recur_limit` | `100` | LangGraph recursion limit |
| `data_vendors.core_stock_apis` | `"yfinance"` | `yfinance` or `alpha_vantage` |
| `data_vendors.technical_indicators` | `"yfinance"` | |
| `data_vendors.fundamental_data` | `"yfinance"` | |
| `data_vendors.news_data` | `"yfinance"` | |
| `tool_vendors` | `{}` | Per-tool vendor overrides (takes precedence over category) |
| `results_dir` | `"./results"` | Output directory |
| `data_cache_dir` | `<package>/dataflows/data_cache` | yfinance cache location |
| `google_thinking_level` | `None` | Google-specific: `"high"` / `"minimal"` |
| `openai_reasoning_effort` | `None` | OpenAI-specific: `"low"` / `"medium"` / `"high"` |
| `anthropic_effort` | `None` | Anthropic-specific: `"low"` / `"medium"` / `"high"` |

---

## 6. Usage

### Python API

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-4-6"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(
    selected_analysts=["market", "news", "fundamentals"],
    debug=True,
    config=config,
)

final_state, decision = ta.propagate("NVDA", "2026-01-15")
# decision: one of BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL

# After observing real-world returns, update agent memories:
ta.reflect_and_remember(returns_losses=500)  # +500 basis points
```

`final_state` is the full `AgentState` dict. All analyst reports, debate transcripts, and decisions are accessible as string values.

### CLI

```bash
pip install .
tradingagents
```

The CLI guides the user through all configuration choices interactively and displays a live multi-pane dashboard during execution.

### Environment Variables

Copy `.env.example` to `.env`. Required keys depend on provider:

| Provider | Required key(s) |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` |
| Ollama | No key (local) |

---

## 7. Extension Points

The architecture is designed around clear seams for common extensions:

### Adding a New Analyst Type

Requires changes in exactly four files — see `.claude/skills/langgraph_agents.md` for the step-by-step pattern:

1. `tradingagents/agents/analysts/` — implement the factory function
2. `tradingagents/graph/trading_graph.py` → `_create_tool_nodes()` — add the tool node
3. `tradingagents/graph/conditional_logic.py` — add the routing method
4. `tradingagents/graph/setup.py` → `setup_graph()` — wire the node into the graph

### Adding a New Data Tool

Requires changes in three places:

1. `tradingagents/dataflows/y_finance.py` (or a new vendor file) — implement the raw fetch function
2. `tradingagents/agents/utils/` — add a `@tool`-decorated wrapper that calls `route_to_vendor()`
3. `tradingagents/dataflows/interface.py` → `VENDOR_METHODS` — register the routing entry

### Adding a New LLM Provider

1. Create `tradingagents/llm_clients/my_provider_client.py` extending `BaseLLMClient`
2. Register it in `create_llm_client()` in `tradingagents/llm_clients/__init__.py` (or `factory.py`)
3. Add provider-specific kwargs extraction to `TradingAgentsGraph._get_provider_kwargs()`

### Modifying Debate Structure

- **Rounds**: Adjust `max_debate_rounds` / `max_risk_discuss_rounds` in config. No code changes needed.
- **New participants**: Add a new agent node and update `ConditionalLogic` to include it in the rotation. The risk debate already demonstrates a 3-way cycle (Aggressive → Conservative → Neutral → Aggressive).
- **New termination condition**: Modify the relevant `should_continue_*` method in `ConditionalLogic`.

---

## 8. Design Decisions & Trade-offs

### BM25 over Embeddings for Memory

**Decision**: `FinancialSituationMemory` uses BM25 lexical matching instead of vector embeddings.

**Rationale**: Avoids a dependency on an embedding API or a local embedding model. Keeps the framework functional offline (with Ollama) and eliminates per-memory-lookup API costs. The trade-off is that BM25 misses semantic similarity when the vocabulary differs — e.g., "quarterly earnings" vs. "Q3 results" — but in financial text with consistent domain vocabulary, lexical overlap tends to be high.

### Stateless Agents (No Inter-Agent Messaging)

**Decision**: Agents communicate exclusively through the shared `AgentState` dict, not via direct messaging.

**Rationale**: LangGraph's execution model is inherently state-machine-based. Direct messaging would require an additional coordination layer. The trade-off is that agents cannot ask each other follow-up questions mid-phase; all inter-agent communication is mediated by the state.

### Two-LLM Architecture (quick vs. deep)

**Decision**: Analysts and debate researchers use a cheaper/faster model; managers (Research Manager, Portfolio Manager) use a more capable model.

**Rationale**: Analysts make many small tool-calling iterations; using an expensive model here multiplies cost by the number of tool calls. Managers make a single high-stakes synthesis decision where quality matters most.

### Sequential Analysts (Not Truly Parallel)

**Decision**: Although described as a "parallel" analyst phase, analysts are currently wired in series in the LangGraph graph (one analyst completes before the next starts).

**Rationale**: LangGraph supports parallel fan-out but requires explicit `Send` API usage. The sequential wiring is simpler and produces the same final state. True parallelism is a natural extension if latency becomes a bottleneck.

### In-Process Memory (No Persistence)

**Decision**: `FinancialSituationMemory` is in-process only; memories are lost on process exit.

**Rationale**: Keeps the core framework dependency-free (no Redis, no database). The `reflect_and_remember()` API is designed so that a persistence layer (Redis, SQLite, etc.) can be added as a drop-in replacement for `FinancialSituationMemory` without changing any agent code.

### String-Based Inter-Agent Outputs

**Decision**: All analyst reports, debate histories, and decisions are plain text strings stored in `AgentState`.

**Rationale**: LLMs consume and produce text naturally. Structured schemas (JSON, dataclasses) would require parsing/validation at every handoff and would reduce the flexibility of agent prompts to evolve the format. The trade-off is that downstream processing (e.g., programmatic extraction of specific metrics) requires parsing unstructured text.

---

## 9. Testing

```bash
# Full test suite
python -m pytest tests/ --tb=short -q

# Single test file
python -m unittest tests.test_ticker_symbol_handling

# Dataflow smoke test (requires network + yfinance)
python test.py
```

The existing test coverage is focused on ticker symbol handling and data format validation. LLM calls and network calls should be mocked in unit tests. Integration tests that hit real APIs should be kept separate and not run in CI.

---

## 10. Known Limitations & Future Directions

| Limitation | Notes |
|---|---|
| **No backtesting harness** | `propagate()` runs a single analysis for a single date. Backtesting requires calling it in a loop and comparing against actual price outcomes. |
| **In-process memory only** | Reflective learning is lost on restart. A Redis or SQLite backend for `FinancialSituationMemory` would make learning persistent across sessions. |
| **yfinance reliability** | Yahoo Finance is rate-limited and frequently changes its undocumented API. Tenacity retry logic mitigates transient failures, but prolonged outages require Alpha Vantage as fallback. |
| **Sequential analysts** | Analyst phase runs analysts in series despite appearing parallel in the conceptual model. True fan-out via LangGraph's `Send` API would reduce end-to-end latency. |
| **News data gaps** | `get_global_news` via yfinance is a partial implementation. Alpha Vantage or a dedicated news API (e.g., NewsAPI, Finnhub) would provide richer macroeconomic coverage. |
| **No position tracking** | The framework produces directional signals (BUY/SELL/HOLD) but has no concept of a portfolio, existing positions, or sizing. Portfolio management is simulated through the debate, not through real position data. |
| **Single-asset analysis** | Each `propagate()` call analyses one ticker in isolation. Cross-asset correlation and portfolio-level optimisation are out of scope. |
