---
name: yfinance_dataflows
description: Patterns for adding new data tools and handling yfinance resiliency in TradingAgents dataflows.
---

# Skill: yfinance Dataflows & Resiliency

## Adding a New Data Tool

New capabilities follow a 3-layer pattern:

### Layer 1 — Implement in `dataflows/` (vendor-specific)

```python
# tradingagents/dataflows/yfinance_utils.py
def get_my_data_yfinance(ticker: str, start_date: str, end_date: str) -> str:
    """Returns markdown-formatted string for LLM consumption."""
    try:
        data = yf.Ticker(ticker).some_property
        if data is None or (hasattr(data, "empty") and data.empty):
            return f"No data available for {ticker}."
        return data.to_markdown()
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"
```

### Layer 2 — Add abstract routing in `agents/utils/agent_utils.py`

```python
@tool
def get_my_data(ticker: Annotated[str, "Stock ticker symbol"], ...) -> str:
    """Tool description shown to the LLM."""
    from tradingagents.dataflows.config import get_config
    config = get_config()
    vendor = config.get("tool_vendors", {}).get(
        "get_my_data",
        config.get("data_vendors", {}).get("my_category", "yfinance")
    )
    if vendor == "yfinance":
        from tradingagents.dataflows.yfinance_utils import get_my_data_yfinance
        return get_my_data_yfinance(ticker, ...)
    raise ValueError(f"Unknown vendor: {vendor}")
```

### Layer 3 — Register in `DEFAULT_CONFIG`

```python
"data_vendors": {
    ...
    "my_category": "yfinance",   # new category
},
```

And add to the appropriate `ToolNode` in `trading_graph.py → _create_tool_nodes`.

---

## Resiliency Rules

### Rate Limit Handling

yfinance returns HTTP 429 or raises vague `ValueError`/JSON decode errors when rate-limited. All download calls must be wrapped:

```python
import time
import random

def fetch_with_backoff(ticker, **kwargs):
    for attempt in range(5):
        try:
            df = yf.download(ticker, threads=False, **kwargs)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(2 ** attempt + random.uniform(0, 1))
    return pd.DataFrame()
```

### Empty DataFrame Guard

Always check `if df.empty:` before any transformation. yfinance silently returns an empty DataFrame for delisted, invalid, or temporarily unavailable tickers:

```python
df = yf.download(ticker, ...)
if df.empty:
    return f"No data available for {ticker}."
```

### Ticker Format

Exchange-qualified tickers (e.g., `7203.T`, `CNQ.TO`) are supported. Always `.upper()` the ticker before passing to yfinance. Never interpolate raw user-supplied ticker strings into Flux/SQL queries without validation.

---

## Tool Output Format

All tool functions must return a **string** — the LLM reads tool results as text. Return markdown tables for tabular data, bullet lists for events. Never return raw Python objects or DataFrames.
