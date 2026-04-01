---
name: langgraph_agents
description: Patterns for adding or modifying LangGraph agent nodes, edges, and state in TradingAgents.
---

# Skill: LangGraph Agent Patterns

## Adding a New Analyst

Adding a new analyst requires changes in **four places** — miss any one and the graph silently skips the analyst.

### 1. Create the agent function (`tradingagents/agents/analysts/`)

```python
def create_my_analyst(llm):
    def my_analyst(state: AgentState) -> dict:
        # Use quick_thinking_llm — analysts are high-frequency nodes
        # Return ONLY the keys you change
        return {"my_report": report_text}
    return my_analyst
```

### 2. Add the tool node (`trading_graph.py` → `_create_tool_nodes`)

```python
"my_analyst": ToolNode([get_my_data_tool])
```

### 3. Register the routing method (`conditional_logic.py`)

```python
def should_continue_my_analyst(self, state: AgentState):
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools_my_analyst"
    return "Msg Clear My_analyst"
```

### 4. Wire into graph (`setup.py` → `setup_graph`)

```python
if "my_analyst" in selected_analysts:
    analyst_nodes["my_analyst"] = create_my_analyst(self.quick_thinking_llm)
    delete_nodes["my_analyst"] = create_msg_delete()
    tool_nodes["my_analyst"] = self.tool_nodes["my_analyst"]
```

The existing sequential loop in `setup_graph` handles chaining automatically once the node is in `analyst_nodes`.

---

## State Field Rules

All `AgentState` fields **must** use `Annotated`:

```python
# CORRECT — LangGraph tracks this field
my_report: Annotated[str, "Report from My Analyst"]

# WRONG — LangGraph ignores updates to this field mid-run
my_report: str
```

Node functions must **return a dict**, never mutate state:

```python
# CORRECT
def my_node(state: AgentState) -> dict:
    return {"my_report": "..."}

# WRONG — mutation is silently ignored by LangGraph
def my_node(state: AgentState) -> dict:
    state["my_report"] = "..."   # Do not do this
    return state
```

---

## Debate Round Pattern

Debates (investment and risk) use a `count` field incremented each round. The conditional router checks `count` against the max rounds config:

```python
def should_continue_debate(self, state: AgentState):
    debate = state["investment_debate_state"]
    if debate["count"] >= self.max_debate_rounds:
        return "Research Manager"
    # Alternate between Bull and Bear
    last_speaker = debate.get("last_speaker", "bear")
    return "Bull Researcher" if last_speaker == "bear" else "Bear Researcher"
```

Always increment `count` in the node that updates the debate state, not in the conditional function.
