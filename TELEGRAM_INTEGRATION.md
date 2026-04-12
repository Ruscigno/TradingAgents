# Telegram Integration via OpenClaw

Run stock analysis, check service health, and manage scheduled reports — all from your phone.

---

## How It Works

```
Telegram (your phone)
      ↕  Telegram Bot API
OpenClaw  (local Mac, Node.js)
      ├─ Telegram channel  ← receives your commands
      ├─ LLM (Ollama / Anthropic)  ← understands and formats results
      └─ MCP server        ← Python subprocess
             ↓  tool calls
TradingAgents MCP Server   (tradingagents/bot/mcp_server.py)
      ├─ run_analysis      → screener + LLM pipeline
      ├─ run_screener      → screener only (fast, no LLM)
      ├─ get_rejected      → why stocks were dropped
      ├─ check_status      → health of all services
      ├─ get_last_result   → cached result without re-running
      └─ manage_schedule   → recurring runs with Telegram push
             ↓
Market Data Service (localhost:8080) → InfluxDB + Redis
```

When you send `/analyze` in Telegram:
1. OpenClaw receives it and forwards to Claude with all tools available.
2. Claude calls `run_analysis()` via the MCP server.
3. The MCP server runs `TechnicalScreener` then `TradingAgentsGraph` for each candidate.
4. Results are returned as structured JSON to Claude.
5. Claude formats one message per passing ticker and sends it back through OpenClaw → Telegram.

---

## Phase 1: Shared Setup (Bot & ID)

Regardless of how you run the bot, you need to create it and find your ID.

### Step 1: Create a Telegram Bot
1. Open Telegram on your phone and search for **@BotFather**.
2. Send `/newbot`.
3. Follow the prompts — choose a name (e.g. `Trading Assistant`) and a username (e.g. `mytrading_bot`).
4. BotFather replies with your **bot token**: `123456789:ABCdef...`
5. Save it — you will need it in the next phase.

### Step 2: Find Your Telegram User ID
The bot only responds to your user ID (single-user security). To find it:
1. Search for **@userinfobot** on Telegram.
2. Send it any message.
3. It replies with your numeric user ID, e.g. `123456789`.

---

## Phase 2: Choose Your Deployment Method

### Option A: Docker (Recommended)

This is the easiest way to run the integration. It avoids local environment conflicts and handles all paths automatically.

#### 1. Configure Environment
Create a file named `.env` in the root of the `TradingAgents` directory:

```bash
# Required for Telegram connection
TELEGRAM_BOT_TOKEN="123456789:ABCdef..."
TELEGRAM_USER_ID="123456789"

# Only needed if using Anthropic instead of Ollama (see LOCAL_LLM_SETUP.md)
# ANTHROPIC_API_KEY="sk-ant-..."
```

#### 2. Start the Services
Ensure `market-data-service` is running, then start the bot:

```bash
docker compose up -d
```

#### 3. Monitor
```bash
docker compose logs -f
```

---

### Option B: Local Manual Setup

Use this only if you cannot run Docker or want to debug the Python code directly on your host.

#### 1. Prerequisites
| Requirement | How to verify |
|-------------|--------------|
| OpenClaw installed | `npm install -g openclaw` |
| Node 22+ | `node --version` |
| Python 3.10+ | `python --version` |
| Ollama running | `ollama list` (see `LOCAL_LLM_SETUP.md`) |
| market-data-service | Running on `localhost:8080` |

#### 2. Environment Variables
Add to your shell profile (`~/.zshrc` or `~/.zprofile`):
```bash
export TELEGRAM_BOT_TOKEN="..."
# Only needed if using Anthropic instead of Ollama (see LOCAL_LLM_SETUP.md)
# export ANTHROPIC_API_KEY="..."
```
Then reload: `source ~/.zshrc`

#### 3. Python Dependencies
```bash
pip install mcp apscheduler
```

#### 4. Configure OpenClaw
OpenClaw reads from `~/.openclaw/openclaw.json5`.
1. `mkdir -p ~/.openclaw`
2. `cp openclaw.config.json5 ~/.openclaw/openclaw.json5`
3. Edit `~/.openclaw/openclaw.json5` and replace `/ABSOLUTE/PATH/TO/TradingAgents` and `YOUR_TELEGRAM_USER_ID` with your actual values.

#### 5. Start OpenClaw
```bash
openclaw gateway
```

---

## Phase 3: Testing the Bot

Open Telegram on your phone, search for your bot by username (e.g. `@mytrading_bot`), and send:

```
/start
```

You should receive a greeting. Then try:

```
/status
```

Expected response:
```
🟢 Market Data Service — healthy (510 tickers)
🟢 InfluxDB            — healthy
🟢 Redis               — healthy (queue: 0, DLQ: 0)
🟢 LLM (Ollama)        — reachable
```

---

## Command Reference

### `/analyze [date] [options]`
Run the full screening + LLM pipeline. Sends one message per passing ticker with the trade decision and rationale.

```
/analyze
/analyze 2026-04-03
/analyze --tickers AAPL NVDA MSFT
/analyze --date 2026-04-03 --rsi-min 40 --max-candidates 10
```

The bot replies immediately with `"Analyzing N candidates..."`, then sends results as they complete. Results are saved to cache (retrievable later with `/last analyze`).

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--date YYYY-MM-DD` | today | Analysis date |
| `--tickers T1 T2 ...` | all MDS tickers | Restrict universe |
| `--rsi-min N` | 50 | RSI lower bound |
| `--rsi-max N` | 70 | RSI upper bound |
| `--vol-osc-min N` | 0 | Volume Oscillator minimum % |
| `--dist-ma-min N` | -5 | Distance from SMA50 lower % |
| `--dist-ma-max N` | 5 | Distance from SMA50 upper % |
| `--max-candidates N` | 20 | LLM pipeline cap |
| `--analysts LIST` | market,news,fundamentals | Comma-separated analyst list |

---

### `/screen [date] [options]`
Run screener only — no LLM calls, returns immediately. Shows pass/fail table with indicator values. Same options as `/analyze`.

```
/screen
/screen 2026-04-03
/screen --tickers AAPL NVDA MSFT TSLA AMZN GOOG
```

Example response:
```
📊 Screener — 2026-04-03  |  3 / 6 passed

Ticker   RSI    VolOsc%  DistMA%  Result
────────────────────────────────────────
AAPL     55.2   +3.1%    -1.2%   ✅ PASS
NVDA     38.0   -24.9%   -8.1%   ❌ RSI 38.0 < 50.0
MSFT     62.1   +0.8%    +0.4%   ✅ PASS
TSLA     71.4   +2.2%    +4.8%   ❌ RSI 71.4 > 70.0
AMZN     59.3   -1.1%    +2.3%   ❌ VolOsc -1.1% ≤ 0.0
GOOG     54.8   +1.4%    -3.2%   ✅ PASS
```

---

### `/rejected [date]`
Show every ticker that failed screening, with the indicator values and the exact filter that rejected it. Uses the cached result from the last `/screen` or `/analyze` for that date, or runs a fresh screen if no cache is available.

```
/rejected
/rejected 2026-04-03
```

Example response:
```
❌ Rejected tickers — 2026-04-03  (487 / 510 failed)

NVDA   RSI=38.0 [need 50–70]   VolOsc=-24.9%   DistMA=-8.1%
       Reason: RSI 38.0 outside [50.0, 70.0]

TSLA   RSI=71.4 [need 50–70]   VolOsc=+2.2%    DistMA=+4.8%
       Reason: RSI 71.4 outside [50.0, 70.0]

AMZN   RSI=59.3                VolOsc=-1.1% [need >0]  DistMA=+2.3%
       Reason: VolOsc -1.1% ≤ 0.0
...
```

---

### `/status`
Check the health of all services. No options.

```
/status
```

Checks:
- **MDS**: `GET /ready` — reports healthy/unhealthy + number of tracked tickers
- **InfluxDB**: via MDS internal health
- **Redis**: ping + job queue depth + DLQ depth
- **LLM API**: test request to confirm key is valid and API is reachable

---

### `/last [analyze|screen]`
Retrieve the last cached result without re-running anything. Instant response.

```
/last analyze
/last screen
/last            ← shows timestamps of both cached results
```

---

### `/schedule`
Manage recurring analysis runs. Scheduled jobs push results to your Telegram chat automatically at the configured time.

**Create a schedule:**
```
/schedule create <recurrence_hours> [start_datetime] [options]
```

| Argument | Description |
|----------|-------------|
| `recurrence_hours` | How often to run (e.g. `24` for daily, `168` for weekly) |
| `start_datetime` | First run time in `YYYY-MM-DDTHH:MM` format (default: now + recurrence) |
| `[options]` | Same options as `/analyze` (--tickers, --rsi-min, etc.) |

Examples:
```
/schedule create 24 2026-04-04T09:00
/schedule create 24 2026-04-07T08:30 --rsi-min 45 --max-candidates 10
/schedule create 168 2026-04-07T09:00 --tickers AAPL NVDA MSFT GOOG
```

The bot responds with a schedule ID:
```
✅ Schedule created  ID: sched_001
   Runs every 24h starting 2026-04-04 09:00
   Next run: 2026-04-04 09:00
```

> **Note:** Each schedule stores its own parameters independently. Changing `/last` or screener.yaml does not affect existing schedules.

**List schedules:**
```
/schedule list
```
```
📅 Active Schedules

ID         Every   Next run              Parameters
──────────────────────────────────────────────────
sched_001  24h     2026-04-04 09:00     default thresholds, all tickers
sched_002  168h    2026-04-07 09:00     --tickers AAPL NVDA MSFT GOOG
```

**Delete a schedule:**
```
/schedule delete sched_001
```

---

## Per-Ticker Screener Thresholds

The bot uses `screener.yaml` in the TradingAgents project root. You can edit it directly to set per-ticker overrides:

```yaml
defaults:
  rsi_min: 50.0
  rsi_max: 70.0
  vol_osc_min: 0.0
  dist_ma_min: -5.0
  dist_ma_max: 5.0

tickers:
  AAPL:
    rsi_min: 45.0
  GOOG:
    rsi_min: 30.0
    rsi_max: 65.0
```

Changes take effect on the next command — no restart needed.

CLI options passed in Telegram (e.g. `/analyze --rsi-min 40`) override the `defaults:` section for that run only, but per-ticker entries in `screener.yaml` always take precedence over both.

---

---

## Troubleshooting & Debugging

### Check Logs
- **Docker**: `docker compose logs -f`
- **Local**: `openclaw logs`

### Bot doesn't respond to messages
- **Docker**: Ensure the container is running: `docker compose ps`. Check logs for connection errors.
- **Local**: Confirm OpenClaw is running: `openclaw status`.
- **Both**: Verify your `TELEGRAM_USER_ID` matches the ID from `@userinfobot` exactly.

### `/status` shows MDS unhealthy
- Ensure `market-data-service` is running.
- **Docker**: The bot connects via `http://api:8080`.
- **Local**: The bot connects via `http://localhost:8080`.

### MCP server fails to start
- **Docker**: This is usually due to a missing environment variable in `.env`. Check `docker compose logs`.
- **Local**:
    - Verify the absolute path in `~/.openclaw/config.json5` is correct.
    - Verify `mcp` and `apscheduler` are installed in your venv: `pip show mcp apscheduler`.
    - Check `PYTHONPATH` points to the TradingAgents directory root.

### Applying Configuration Changes
If you modify `screener.yaml`, changes take effect immediately on the next command.
If you modify `openclaw.config.json5` (Local) or `.env` (Docker):
- **Docker**: `docker compose restart`
- **Local**: `openclaw gateway restart`

---

## Technical Appendix: Manual MCP Testing

You can test the MCP server or the analysis tools directly (useful for developers):

```bash
# Run the MCP server in your terminal
python tradingagents/bot/mcp_server.py

# Test a specific tool in a Python shell
python -c "from tradingagents.bot.analysis_tools import check_status; import json; print(json.dumps(check_status(), indent=2))"
```
