# 01 — Cascata Eliminatória de Recomendações de Trade

> **Status:** proposta de design — revisão 2 (decisões do Sander incorporadas)
> **Autor:** Claude (sob direção de Sander)
> **Data:** 2026-04-27

## Contexto

O TradingAgents já roda hoje uma cascata multi-agente bem definida (analistas → debate de pesquisa → trader → debate de risco → portfolio manager) **para um único ticker por vez**, invocada via `python -m cli.main analyze` ([`cli/main.py`](../cli/main.py)). A cascata é ótima como mecanismo de análise profunda, mas não responde à pergunta natural do dia-a-dia:

> *"Dada uma lista de N ações candidatas, quais são as melhores oportunidades de trade hoje?"*

Rodar a cascata completa em N ações é caro, lento e desperdiça tokens em candidatas que dariam para descartar com filtros mais baratos.

Este estudo propõe um **novo comando** (`recommend` ou similar) que percorre as etapas de filtro **da mais barata para a mais cara**, eliminando candidatas em cada etapa, e que **encerra cedo** quando não sobra nenhuma. Reaproveita os agentes existentes — não os reescreve.

---

## Decisões já tomadas (revisão 2)

Estas escolhas estão fechadas e moldam o resto do documento:

| # | Decisão | Implicação |
|---|---|---|
| **D0** | **Todos os agentes que usam LLM rodam Kimi K2.6 via OpenRouter** ($0.95 / $4 por M tokens) — em vez do `gpt-5-mini` / `gpt-5.2` do default atual. GPT-5 mata o ROI antes do sistema gerar dinheiro. | Trocar `default_config.py` na branch desta feature, ou parametrizar via `--llm-profile recommend`. |
| **D1** | **Universo vem do MCP do market-data-service** (já registrado como MCP server, expõe `list_tickers`). Watchlist em arquivo e screeners externos ficam fora desta primeira versão. | Etapa 0 chama o MCP, não lê arquivo. |
| **D2** | **Cada etapa emite verdict explícito `PASS` ou `NO_PASS` por ticker** + uma `confidence ∈ [0,1]`. Etapas com LLM podem usar o LLM para decidir o veredito (com prompt estruturado retornando JSON). | Schema do `StageResult` muda: era `passed`/`eliminated`, vira lista única de candidatas todas com `verdict` carimbado. |
| **D3** | **Sem top-K por etapa.** Todos que passarem do threshold avançam. **Mas o resultado final é ordenado por probabilidade**, via score acumulado (proposta de mecanismo abaixo). | Mais candidatas chegam às etapas finais → custo varia mais com a qualidade do dia. Skip-on-zero compensa. |
| **D4** | **Position sizing fixo** na primeira versão (ex: 2% do capital por trade). Sem cálculo dinâmico de Kelly, vol-targeting, etc. | Etapa 6 só decide `BUY` / `SELL` / `HOLD` — tamanho vem da config. |
| **D5** | **Memória dos researchers ligada por default.** Roda 1× por dia; consultas avulsas no mesmo dia se beneficiam do contexto acumulado. | Etapas 4 e 5 leem [`agents/utils/memory.py`](../tradingagents/agents/utils/memory.py); recommend.yaml: `memory.enabled: true`. |
| **D6** | **Paralelismo: `max_workers = 4`** dentro de cada etapa (4 tickers em paralelo). Conservador frente ao rate-limit do OpenRouter. | Configurável; ajusta-se se observar 429s nos logs. |
| **D7** | **Cadência: 1× por dia útil, ~15 min após o close do regular session do NYSE** (16:30 ET). Pula sábados, domingos e feriados. Resultado vai automaticamente pro **Telegram**. | Cron em Hermes; calendário `XNYS` via `exchange_calendars` (mesma fonte de verdade do market-data-service); entrega via bot existente. Detalhamento na seção "Cadência e entrega". |

### Mecanismo de scoring acumulado (proposta — D3)

Cada candidata carrega um **histórico de confidences** das etapas que rodaram:

```python
candidate.confidence_history = {
  "screener":  0.72,
  "market":    0.81,
  "analysts":  0.66,
  "debate":    0.78,
  "research":  0.85,
  "decision":  0.90,
}
```

O **score final** ordena o resultado e é **média ponderada com pesos crescentes** (etapas mais profundas pesam mais, porque viram mais informação):

```python
weights = {
  "screener":  0.05,
  "market":    0.10,
  "analysts":  0.15,
  "debate":    0.20,
  "research":  0.25,
  "decision":  0.25,   # soma = 1.0
}
final_score = sum(weights[s] * confidence_history[s] for s in confidence_history)
```

Vantagens:
- Interpretável (entre 0 e 1, comparável entre tickers).
- Pesos parametrizáveis (`config.yaml`) → fácil recalibrar.
- Stages que não rodaram (skip-on-zero não é o caso aqui — uma candidata só "não roda" uma etapa se foi eliminada antes; nesse caso ela já não está no resultado final).
- Tiebreak natural: confidence da etapa mais profunda que rodou.

**Alternativas consideradas e rejeitadas:**
- *Produto das confidences* (interpretação como probabilidade conjunta): supõe independência condicional entre etapas, o que é falso (passar na etapa 4 prevê passar na 5). Rejeitado.
- *Só a confidence da última etapa*: descarta info das anteriores. Rejeitado por desperdiçar sinal.
- *Top-K rígido*: explicitamente excluído pelo usuário.

---

## Estado atual em uma página

| Componente | Onde | Modelo | Custo relativo |
|---|---|---|---|
| Market Analyst | [`agents/analysts/`](../tradingagents/agents/analysts/) | `gpt-5-mini` | barato |
| Social Media Analyst | idem | `gpt-5-mini` | barato |
| News Analyst | idem | `gpt-5-mini` | barato |
| Fundamentals Analyst | idem | `gpt-5-mini` | barato |
| Bull / Bear Researchers | [`agents/researchers/`](../tradingagents/agents/researchers/) | `gpt-5-mini` × `max_debate_rounds` | médio (multi-turn) |
| Research Manager | [`agents/managers/`](../tradingagents/agents/managers/) | **`gpt-5.2`** | **caro** |
| Trader | [`agents/trader/`](../tradingagents/agents/trader/) | `gpt-5-mini` | barato |
| Aggressive / Neutral / Conservative Debators | [`agents/risk_mgmt/`](../tradingagents/agents/risk_mgmt/) | `gpt-5-mini` × `max_risk_discuss_rounds` | médio (multi-turn) |
| Portfolio Manager | [`agents/managers/`](../tradingagents/agents/managers/) | **`gpt-5.2`** | **caro** |

Outros achados relevantes:

- **Screener técnico já existe** mas **não é usado** pela CLI: [`tradingagents/screener/technical_screener.py`](../tradingagents/screener/technical_screener.py) (RSI + Volume Oscillator + distância do SMA50). Filtro determinístico, custo zero — candidato natural a primeira etapa.
- **Universo de tickers**: hoje só existe entrada manual (`typer.prompt`, default `SPY`) em [`cli/main.py:604`](../cli/main.py#L604). Não há watchlist, nem batch mode.
- **State LangGraph**: tudo passa em memória via `AgentState` ([`agents/utils/agent_states.py`](../tradingagents/agents/utils/agent_states.py)). Saídas vão para JSON e markdown em `eval_results/{ticker}/...`.
- **Defaults de modelo**: [`tradingagents/default_config.py:11`](../tradingagents/default_config.py#L11) — `quick_think_llm = "gpt-5-mini"`, `deep_think_llm = "gpt-5.2"`.

---

## Princípios da cascata

1. **Eliminatório.** Cada etapa só recebe candidatas que sobreviveram à anterior. Não é "ranking" no fim — é peneira incremental.
2. **Barato → caro.** Etapas usam modelos/recursos progressivamente mais custosos. O filtro mais agressivo ($0) vem primeiro; o LLM de raciocínio profundo ($$$) só vê os top N.
3. **Skip-on-zero.** Se uma etapa retorna 0 candidatas, as etapas seguintes **não rodam**. Skip também se sobrar 1 e a etapa só agregar valor com comparação relativa.
4. **Determinístico onde possível.** Stages 0 e 1 são puras (pandas, sem LLM) — reproduzíveis e auditáveis sem chave de API.
5. **Auditável.** Cada eliminação grava ticker + etapa + motivo + score. Fácil revisar "por que ABCD foi descartado".
6. **Reutilizar agentes existentes**, não reescrever. As únicas peças novas são: orquestrador da cascata, gates entre etapas, persistência das eliminações.

---

## Etapas propostas

Os números entre parênteses são chutes de ordem de magnitude para um universo inicial de **N=50 tickers** num dia típico — refinar depois com dados reais.

### Etapa 0 — Seleção do universo (custo $0, instantâneo)

**Entrada:** nada (configuração).
**Saída:** lista de tickers candidatos (qualquer N — depende do que o market-data-service rastreia).

**Fonte:** chamar a tool MCP `list_tickers` do `market-data-service` (já registrado e testado neste workspace). Retorna a lista completa de tickers com last-fetch timestamp para 1m e 1d. Eliminamos imediatamente:

- Tickers com `last_fetch_1d` mais antigo que `today - 5 dias` (dados estagnados).
- Tickers sem dados 1d (`last_fetch_1d == None`).

Implementação: o orquestrador instancia um cliente MCP simples (HTTP streamable), faz `list_tickers`, parseia.

**Por que não watchlist em arquivo:** decisão D1. Único universo é o que o market-data-service rastreia.

**Verdict:** N/A (etapa de seleção, não de filtro). Não emite confidence.

### Etapa 1 — Screener técnico determinístico (custo $0, ~segundos)

**Entrada:** ~50 tickers.
**Saída:** subconjunto que passa nos critérios técnicos.
**Mecanismo:** [`tradingagents/screener/technical_screener.py`](../tradingagents/screener/technical_screener.py) já implementa RSI + Volume Oscillator + distância do SMA50.

**Critérios de PASS (configuráveis em `recommend.yaml`):**
- `30 ≤ RSI ≤ 70` (foge de extremos saturados)
- Volume oscillator > 0 (volume atual acima do MA recente — confirmação)
- |preço − SMA50| / SMA50 ≤ 0.05 (preço dentro de ±5% do SMA50, não muito esticado)

Falhar em qualquer um → `verdict = NO_PASS`.

**Confidence quando PASS:** mapear cada indicador para [0,1] e fazer média:
```python
c_rsi    = 1 - abs(RSI - 50) / 20    # 1.0 quando RSI=50, 0.0 quando RSI=70 ou 30
c_vol    = min(volume_oscillator / 30, 1.0)
c_sma    = 1 - distancia_sma50 / 0.05
confidence = (c_rsi + c_vol + c_sma) / 3
```

**Por que primeiro:** custo zero, sem LLM, derruba a maior parte do ruído em segundos.

### Etapa 2 — Market Analyst isolado (custo $, ~10s/ticker)

**Entrada:** sobreviventes da etapa 1.
**Modelo LLM:** **Kimi K2.6** via OpenRouter.
**Mecanismo:** rodar **apenas o Market Analyst** ([`agents/analysts/market_analyst.py`](../tradingagents/agents/analysts/market_analyst.py)) por ticker, com prompt augmentado pra retornar **JSON estruturado** com veredito explícito:

```json
{
  "verdict": "PASS" | "NO_PASS",
  "confidence": 0.0..1.0,
  "direction": "long" | "short" | "none",
  "narrative": "..."
}
```

**Prompt addendum (resumo):** *"...com base nos indicadores técnicos acima, esta ação tem setup técnico para entrada hoje? Responda em JSON com `verdict` (PASS ou NO_PASS), `confidence` (0 a 1, sua certeza no veredito), `direction` (long, short, ou none), e `narrative` (1 parágrafo)."*

**Por que Market Analyst isolado:** os 4 analistas hoje rodam em paralelo, mas só Market Analyst usa indicadores técnicos puros — os outros (Sentiment, News, Fundamentals) tocam APIs externas (FinnHub, Tavily) e fazem reasoning sobre texto. Esta etapa é a primeira onde gastamos token: queremos o filtro LLM mais barato. Os outros 3 analistas só rodam pra quem passar daqui.

**Custo estimado por ticker:** ~$0.005 (Kimi K2.6, prompt ~3k tokens, output ~500 tokens).

### Etapa 3 — Demais analistas + agregação (custo $$, ~1min/ticker)

**Entrada:** sobreviventes da etapa 2.
**Modelo LLM:** **Kimi K2.6** (todos os 3 analistas).
**Mecanismo:** rodar Social Media + News + Fundamentals em paralelo. Cada um retorna seu próprio JSON estruturado (mesmo schema da etapa 2). Depois um **agregador determinístico** combina:

```python
composite_confidence = (
    w_market     * conf_market     +
    w_sentiment  * conf_sentiment  +
    w_news       * conf_news       +
    w_fundamental* conf_fundamental
)  # sum of weights = 1
```

**Verdict da etapa = PASS se:**
- `composite_confidence ≥ 0.65` (configurável), **E**
- Nenhum dos 3 novos analistas retornou `NO_PASS` com `confidence ≥ 0.85` (veto: alta certeza de "não passa" em qualquer dimensão derruba a candidata, mesmo com média alta).

A regra de veto é importante: um earnings miss flagrante ou notícia regulatória negativa derruba mesmo com técnico bom.

**`confidence` reportada para o histórico:** `composite_confidence`.

**Custo estimado por ticker:** ~$0.015 (3 LLM calls + 1 chamada Tavily/FinnHub).

### Etapa 4 — Debate Bull × Bear (custo $$$, ~2min/ticker)

**Entrada:** sobreviventes da etapa 3.
**Modelo LLM:** **Kimi K2.6** (Bull, Bear e o judge).
**Mecanismo:** debate existente ([`agents/researchers/`](../tradingagents/agents/researchers/)), com `max_debate_rounds = 1` para a fase de filtro (em vez do default 2). Após o debate, um **judge call** (Kimi K2.6 também — não usa "deep LLM") emite veredito estruturado:

```json
{
  "verdict": "PASS" | "NO_PASS",
  "confidence": 0.0..1.0,
  "direction": "long" | "short",
  "rationale": "..."
}
```

`verdict = NO_PASS` significa que o judge não viu tese convincente em nenhuma das direções (bull e bear ambos fracos).

**Threshold de PASS:** `confidence ≥ 0.7`.

**Custo:** ~$0.014/ticker (Kimi K2.6, ~1 round = 2 turns + 1 judge = ~6k input + 2k output).

### Etapa 5 — Research Manager (custo $$$$, ~3min/ticker)

**Entrada:** sobreviventes da etapa 4.
**Modelo LLM:** **Kimi K2.6** (decisão D0: substituímos `gpt-5.2` por Kimi mesmo no "deep think" — mantém a qualidade alta, custo radicalmente menor).
**Mecanismo:** Research Manager existente ([`agents/managers/research_manager.py`](../tradingagents/agents/managers/research_manager.py)) gera o plano de investimento consolidado a partir de todos os reports anteriores.

Ainda assim **emite verdict** mesmo sendo a etapa "convergente": se ao revisar todos os sinais o plano consolidado ficar contraditório (e.g., técnico long mas fundamentais negativos não considerados antes), pode emitir `NO_PASS`. Ou seja, a etapa **pode** descartar — embora isso seja raro.

JSON de saída:

```json
{
  "verdict": "PASS" | "NO_PASS",
  "confidence": 0.0..1.0,
  "investment_plan": "...",     // markdown longo
  "key_risks": ["...", "..."]
}
```

**Threshold de PASS:** `confidence ≥ 0.7`.

**Skip-on-zero:** se etapa 4 não passou ninguém, etapa 5 não roda.

**Custo estimado por ticker:** ~$0.018 (Kimi K2.6, ~10k input + 2k output — contexto grande agregando todos os reports).

### Etapa 6 — Trader + Risk debate + Portfolio Manager (custo $$$$$, ~5min/ticker)

**Entrada:** sobreviventes da etapa 5.
**Modelo LLM:** **Kimi K2.6** (todos os agentes desta etapa).
**Saída final por ticker:** decisão estruturada:

```json
{
  "verdict": "PASS" | "NO_PASS",
  "confidence": 0.0..1.0,
  "action": "BUY" | "SELL" | "HOLD",
  "stop_loss": 178.50,
  "take_profit": 195.00,
  "rationale": "...",
  "position_size_pct": 2.0       // FIXO via config — decisão D4
}
```

`position_size_pct` **vem da config**, não é calculado por LLM (decisão D4). Default: 2% do capital por trade. Limite de trades simultâneos também por config (default: 5).

**Mecanismo:** stage final do graph atual — Trader executa o plano da etapa 5, Risk Debate roda com 3 perspectivas (Aggressive/Neutral/Conservative, `max_risk_discuss_rounds = 1` para filtro, default 2 para análise full), Portfolio Manager emite veredito final.

**Skip total se etapa 5 retornou 0.**

**Threshold de PASS:** `confidence ≥ 0.7` E `action ∈ {BUY, SELL}` (HOLD vira NO_PASS porque não é trade acionável).

**Custo estimado por ticker:** ~$0.031 (Kimi K2.6, multi-turn debate + 2 chamadas longas — ~20k input + 3k output).

---

## Custos comparativos com Kimi K2.6 (decisão D0)

Preço Kimi K2.6 via OpenRouter: **$0.95 / M tokens input**, **$4 / M tokens output**.

**Custo por ticker em cada etapa (estimativa):**

| Etapa | Input médio | Output médio | $/ticker |
|---|---:|---:|---:|
| 0. Universe | — | — | $0 |
| 1. Screener | — | — | $0 |
| 2. Market Analyst | 3k | 0.5k | $0.005 |
| 3. Demais analistas (3×) + agg | 9k | 1.5k | $0.014 |
| 4. Bull×Bear debate + judge | 6k | 2k | $0.014 |
| 5. Research Manager | 10k | 2k | $0.018 |
| 6. Trader + Risk + Portfolio | 20k | 3k | $0.031 |

**Cenários (universo N=50, funil 50→12→5→3→2→1):**

| Cenário | Etapas | Custo |
|---|---|---:|
| **Cascata eliminatória, dia típico** | 0–6 afunilando | **~$0.23 / run** |
| **Cascata eliminatória, dia ruim (etapa 1 zera)** | 0–1 | **$0** |
| **Cascata eliminatória, dia ótimo (todas avançam)** | todas em todos | ~$4.10 / run |
| Cascata atual com Kimi (sem filtros, todos os tickers) | 6 em todos | ~$4.10 |
| **Cascata atual com gpt-5 (sem filtros)** | 6 em todos | **~$50–150** |

**Conclusões:**
- Kimi sozinho já elimina o problema de custo (~25× mais barato que GPT-5). A cascata eliminatória adiciona mais ~20× de redução em dia típico.
- Combinado: ~500× mais barato que rodar a cascata original com GPT-5 em 50 tickers.
- A esses preços, dá pra rodar a cada hora sem culpa. Roda 24×/dia = ~$5.50/dia em dias típicos.

---

## Implementação proposta

### Novo comando Typer

Em [`cli/main.py`](../cli/main.py), adicionar:

```python
@app.command()
def recommend(
    config_path: Path = typer.Option(
        Path("tradingagents/recommend/recommend.yaml"),
        "--config", "-c",
        help="YAML com thresholds, pesos e endpoint MCP",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Roda só etapas 0-1 (deterministas), estima custo das LLM stages",
    ),
    output_format: str = typer.Option(
        "table", "--format",
        help="table | json | markdown — formato do resultado final ordenado",
    ),
    max_workers: int = typer.Option(
        4, "--workers",
        help="Paralelismo dentro de cada etapa (cuidado com rate-limit OpenRouter)",
    ),
):
    """Cascata eliminatória de recomendações: barato → caro, skip-on-zero,
    universo do market-data-service via MCP."""
    run_recommend(...)
```

Removidas as flags antigas:
- `--universe-file` / `--from-mcp` → universo agora é fixo MCP (D1).
- `--max-per-stage` → todos acima do threshold avançam (D3).

### Orquestrador

Novo módulo [`tradingagents/recommend/`](../tradingagents/recommend/):

```
recommend/
├── __init__.py
├── pipeline.py          # roda as 6 etapas com gates
├── stages.py            # StageResult, Stage protocol
├── stage_universe.py    # etapa 0
├── stage_screener.py    # etapa 1 (wrapper do technical_screener.py existente)
├── stage_market.py      # etapa 2
├── stage_analysts.py    # etapa 3
├── stage_debate.py      # etapa 4
├── stage_research.py    # etapa 5
├── stage_decision.py    # etapa 6 (chama o graph existente sem reanalysis)
└── persistence.py       # grava run + eliminações
```

Pseudocódigo do orquestrador:

```python
WEIGHTS = {  # decisão D3 — pesos crescentes para etapas mais profundas
    "screener": 0.05, "market":   0.10, "analysts": 0.15,
    "debate":   0.20, "research": 0.25, "decision": 0.25,
}

def run_recommend(config) -> list[Candidate]:
    universe = stage_universe.run(config)       # MCP list_tickers
    if not universe: return []

    survivors = universe
    for stage in [stage_screener, stage_market, stage_analysts,
                  stage_debate,   stage_research, stage_decision]:
        if not survivors:
            log_skip(stage.name, reason="zero candidatos — encerra cascata")
            break

        result = stage.run(survivors, config)   # roda LLM/lógica em todos
        survivors = [c for c in result if c.verdict == "PASS"]
        eliminated = [c for c in result if c.verdict == "NO_PASS"]

        log_run(stage.name,
                before=len(result), after=len(survivors),
                eliminated=eliminated, cost=stage.last_cost)

    # Score final = média ponderada das confidences acumuladas
    for c in survivors:
        c.final_score = sum(
            WEIGHTS[s] * c.confidence_history[s]
            for s in c.confidence_history
        )

    return sorted(survivors, key=lambda c: c.final_score, reverse=True)
```

### Schema do contrato entre etapas

```python
@dataclass
class Candidate:
    ticker: str
    confidence_history: dict[str, float]  # por stage_name → confidence ∈ [0,1]
    reports: dict[str, str]               # outputs textuais (passados às etapas seguintes)
    elimination: Optional[Elimination] = None  # se NO_PASS em alguma etapa
    final_score: Optional[float] = None        # preenchido após etapa final

@dataclass
class Elimination:
    stage: str               # "screener", "market", "analysts", ...
    reason: str              # texto humanamente legível
    confidence: float        # confidence reportada quando NO_PASS

@dataclass
class StageOutcome:           # output de stage.run()
    candidates: list[Candidate]   # todos com verdict carimbado
    cost_usd: float

@dataclass
class Decision:               # candidatas que sobreviveram (subset com action)
    ticker: str
    final_score: float
    action: Literal["BUY", "SELL"]
    stop_loss: float
    take_profit: float
    position_size_pct: float    # constante via config (D4)
    confidence_history: dict[str, float]
    rationale: str
```

### Verdict de cada etapa (resumo)

| Etapa | `PASS` quando |
|---|---|
| 1. screener | RSI ∈ [30,70] **E** vol_osc > 0 **E** \|preço−SMA50\|/SMA50 ≤ 0.05 |
| 2. market | LLM retorna `verdict=PASS` (estruturado em JSON) |
| 3. analysts | composite ≥ 0.65 **E** nenhum analista deu NO_PASS com conf ≥ 0.85 |
| 4. debate | judge retorna `verdict=PASS` com confidence ≥ 0.7 |
| 5. research | research manager retorna `verdict=PASS` com confidence ≥ 0.7 |
| 6. decision | portfolio manager retorna `verdict=PASS` com `action ∈ {BUY,SELL}` e confidence ≥ 0.7 |

### Persistência

Cada execução cria `eval_results/_recommend_runs/{date}_{run_id}.json`:

```json
{
  "started_at": "2026-04-27T18:00:00Z",
  "universe_source": "mcp:market-data-service",
  "universe_size": 50,
  "llm_provider": "openrouter",
  "llm_model": "moonshotai/kimi-k2.6",
  "stages": [
    {
      "name": "screener",
      "before": 50, "passed": 12,
      "eliminated": [
        {"ticker": "XYZ", "reason": "RSI=82 (>70)", "confidence": null}
      ],
      "duration_s": 1.2,
      "cost_usd": 0.0
    },
    {
      "name": "market",
      "before": 12, "passed": 7,
      "eliminated": [
        {"ticker": "ABC", "reason": "Setup técnico fraco; price action lateral",
         "confidence": 0.34}
      ],
      "duration_s": 28.5,
      "cost_usd": 0.062
    }
    // ... uma entrada por etapa que rodou
  ],
  "skipped_stages": [],   // preenchido se etapa anterior zerou
  "decisions": [
    {
      "ticker": "AAPL",
      "final_score": 0.823,                   // ordenação primária
      "action": "BUY",
      "position_size_pct": 2.0,
      "stop_loss": 178.5,
      "take_profit": 195.0,
      "rationale": "...",
      "confidence_history": {
        "screener": 0.85, "market": 0.82, "analysts": 0.78,
        "debate": 0.80,   "research": 0.84, "decision": 0.86
      }
    }
  ],
  "total_cost_usd": 0.234
}
```

Fica fácil revisar 30 dias depois: "por que NVDA não passou da etapa 3 em 22/04?" → grep + jq, leitura humana direta.

### Configuração

Novo arquivo `tradingagents/recommend/recommend.yaml.example`:

```yaml
# LLM provider (decisão D0)
llm:
  provider: openrouter
  model: moonshotai/kimi-k2.6
  api_key_env: OPENROUTER_API_KEY
  base_url: https://openrouter.ai/api/v1
  temperature: 0      # determinismo

# Universo (decisão D1) — única fonte: MCP do market-data-service
universe:
  source: mcp
  mcp_url: http://localhost:8080/mcp
  filter:
    require_recent_1d: true
    max_staleness_days: 5

# Thresholds por etapa (decisão D2 — verdict PASS/NO_PASS)
thresholds:
  screener_indicators_required: all   # todos os 3 critérios
  market_confidence_min: 0.6
  analysts_composite_min: 0.65
  analysts_veto_threshold: 0.85       # NO_PASS forte de qq dimensão derruba
  debate_confidence_min: 0.7
  research_confidence_min: 0.7
  decision_confidence_min: 0.7

# Pesos da composição da etapa 3 (somam 1)
analyst_weights:
  market:      0.35
  sentiment:   0.15
  news:        0.20
  fundamental: 0.30

# Pesos do score final ordenador (decisão D3 — somam 1)
final_score_weights:
  screener: 0.05
  market:   0.10
  analysts: 0.15
  debate:   0.20
  research: 0.25
  decision: 0.25

# Position sizing fixo (decisão D4)
position:
  size_pct: 2.0
  max_concurrent_trades: 5

# Tuning de debate
debate:
  rounds_filter: 1   # rounds reduzidos durante etapa 4
  rounds_final: 2    # rounds completos no estágio final (etapa 6 risk debate)
```

---

## Cadência e entrega via Telegram (decisão D7)

### Quando rodar

- **Frequência:** 1× por dia útil do mercado americano (NYSE).
- **Horário:** **16:30 ET** (15 min após o close regular das 16:00 ET). Buffer de 15 min permite que os preços de fechamento se assentem nos feeds (Yahoo Finance, etc.) e o market-data-service tenha tempo de ingerir o close do dia.
- **Half-days** (ex: dia depois do Thanksgiving, véspera de Natal): NYSE fecha 13:00 ET. O cron pode rodar no mesmo horário (16:30 ET) — não precisa caso especial: às 16:30 o mercado já fechou de qualquer jeito.

### Calendário (skip de fim de semana e feriados)

A fonte de verdade é **`exchange_calendars.get_calendar("XNYS")`** — mesmo calendário usado pelo market-data-service ([CLAUDE.md do market-data-service](../../market-data-service/CLAUDE.md#L40)). Adicionar como dependência do TradingAgents:

```toml
# pyproject.toml ou requirements.txt
exchange-calendars >= 4.2
```

Lógica de gate na entrada do comando (executa imediatamente; sai com exit 0 se for não-trading-day, sem rodar nada):

```python
import exchange_calendars as xcals
from datetime import datetime, timezone

def is_trading_day(ts: datetime | None = None) -> bool:
    cal = xcals.get_calendar("XNYS")
    ts = ts or datetime.now(tz=timezone.utc)
    return cal.is_session(ts.astimezone(cal.tz).strftime("%Y-%m-%d"))

# No início do recommend(), antes da etapa 0:
if not is_trading_day():
    log.info("Mercado fechado hoje (weekend/feriado XNYS) — pulando run.")
    raise typer.Exit(0)
```

### Onde mora o cron

O cron deve viver num **container Hermes deste projeto** (TradingAgents), espelhando o padrão já usado pelo market-data-service ([`hermes/`](../../market-data-service/hermes/)). Hermes tem cron nativo (`hermes cron`) e bot Telegram já integrado.

**Estrutura sugerida** (estudo separado para detalhar):

```
~/projects/TradingAgents/hermes/
├── docker-compose.yml      # imagem reutilizada market-data-service/hermes-agent:2026.4.23
├── .env                    # bot token + OPENROUTER_API_KEY (gitignored)
├── config.yaml             # MCP do market-data-service via host.docker.internal:8080
├── cron/
│   └── daily-recommend.md  # prompt do cron
└── data/...
```

O **prompt do cron** (em `cron/daily-recommend.md`) invoca a CLI do TradingAgents (montada como volume ou rebuilt na imagem) e formata a saída pra Telegram:

```markdown
[SYSTEM: cron job, market close. Rode `python -m cli.main recommend --format json`,
parseie o resultado, envie para o canal home no Telegram.]

Use a tool `shell` para executar:
  cd /workspace && python -m cli.main recommend --format json

Quando obtiver o JSON com `decisions`, formate em mensagem markdown como:

🔔 *Recomendações de trade — {data}*
Universo: {universe_size} → final: {len(decisions)}
Custo do run: ${total_cost_usd:.3f}

| # | Ticker | Action | Score | Stop | Target |
|---|--------|--------|------:|-----:|-------:|
{loop sobre decisions ordenadas por final_score desc}

Envie para o canal home.
```

### Bot Telegram para entrega

**Question aberta (decidir antes do deployment):** o `@sander_trading_bot` (que está hoje no market-data-service-hermes) cabe melhor semanticamente no TradingAgents-hermes — é literalmente um bot de trading.

**Sugestão:**
- Mover o `@sander_trading_bot` pro TradingAgents-hermes (este projeto).
- Criar um novo bot para market-data-service (ex: `@sander_marketdata_bot`) — usado para queries de dados ad hoc.
- Ou: usar o mesmo bot, mas só uma das instâncias polando (a do TradingAgents). A do market-data-service desliga o gateway Telegram.

Vou tratar essa migração num estudo separado quando formos implementar o cron — não bloqueia este estudo.

### Formato da mensagem entregue

Mockup do que cai no Telegram após o close:

```
🔔 Recomendações de trade — 2026-04-28 (Mon)
Universo: 50 tickers → 3 finais
Custo do run: $0.182
Tempo: 4m 12s

🟢 BUY  AAPL  score=0.86  stop=178.50  target=195.00
🟢 BUY  MSFT  score=0.79  stop=412.00  target=440.00
🔴 SELL TSLA  score=0.72  stop=180.00  target=160.00

Position size: 2% por trade (config)
Detalhes completos: eval_results/_recommend_runs/2026-04-28_*.json
```

Caso 0 candidatas finais:
```
ℹ️ Recomendações de trade — 2026-04-28 (Mon)
Universo: 50 tickers — nenhuma sobreviveu até a etapa final.
Etapa de saída: 4 (debate Bull×Bear) — 0 com confidence ≥ 0.7.
Custo do run: $0.094
```

Caso dia não-útil:
```
(o cron simplesmente não roda — Hermes cron respeita o gate de calendar)
```

### Recapitulação do gate

Ordem de checagens no início do `recommend`:

1. `is_trading_day()` — XNYS calendar — se False, exit 0 sem entrar na cascata.
2. Hora local correta (ex: rejeitar runs antes de 16:30 ET com flag `--force` para override).
3. market-data-service api alcançável (test `/health`) — se não, sai com erro e Telegram avisa.
4. Etapa 0 → universo via MCP.

---

## Auditabilidade e reprodutibilidade

- **Determinístico até onde dá**: etapas 0 e 1 são reproduzíveis bit-a-bit. Etapas 2–6 com LLM são determinísticas se `temperature=0`.
- **Eliminações documentadas**: cada candidata descartada vai pro JSON com motivo legível.
- **Custos rastreados**: cada etapa estima custo antes (via tokenizer) e mede depois (via response usage).
- **Reentrante**: se a execução crashar na etapa 4, o estado das etapas 0–3 está em disco — pode retomar.

---

## Triggers (como o sistema é invocado)

1. **Cron diário (caminho principal — D7):** Hermes cron em container deste projeto, dispara 16:30 ET nos dias úteis (gated por XNYS calendar). Resultado vai pro Telegram. Detalhe na seção "Cadência e entrega".
2. **CLI direto (manual / debug):**
   ```bash
   python -m cli.main recommend                    # roda com config default
   python -m cli.main recommend --dry-run          # sem LLM, só etapas 0-1
   python -m cli.main recommend --format json      # output estruturado, ideal pro cron
   ```
3. **Telegram chat ad hoc:** usuário pergunta no chat *"como ficaram as recomendações de hoje?"* — Hermes lê o último `eval_results/_recommend_runs/{today}_*.json` e responde resumindo. Não dispara nova run (custo). Beneficia-se da memória D5.

---

## Perguntas em aberto

Todas as decisões de produto estão tomadas (D1–D7). Sobram só pontos de implementação que ficam mais claros enquanto se constrói:

1. **Container Hermes do TradingAgents.** Estudo separado para detalhar o setup (paralelo ao `~/projects/market-data-service/hermes/`). Decisões pendentes: bot Telegram (mover `@sander_trading_bot` ou criar novo `@sander_recommend_bot`?), e como TradingAgents code chega no container (mount do projeto vs rebuild da imagem com pip install).
2. **Calibração inicial dos thresholds** (`market_confidence_min`, `analysts_composite_min`, etc.). Os defaults propostos (0.6–0.7) são chutes razoáveis. Refinar depois das primeiras 10–20 runs reais — muito alto = não passa nada; muito baixo = entrega ruído.
3. **Estratégia para dias half-day** (early close 13:00 ET). O cron 16:30 ET cobre, mas talvez queira rodar logo após o early close (13:30 ET) nesses dias. Detalhe operacional, decidir quando observar o primeiro half-day após o lançamento.

---

## Próximos passos sugeridos (ordem)

1. **Adicionar `exchange-calendars` à dependência** + função `is_trading_day()` num utils novo. Test simples. ~20min.
2. **Cliente MCP** + etapa 0 (chamada `list_tickers` ao market-data-service). ~1h.
3. **Etapa 1** (wrapper sobre `technical_screener.py` existente, normaliza saída pro novo `Candidate`). ~1h.
4. **Smoke test `--dry-run`**: confere `is_trading_day` → universe (MCP) → screener — sem LLM, custo $0.
5. **Etapa 2** (Market Analyst com prompt JSON estruturado, parser de verdict, retry se JSON inválido). ~2h.
6. **Gate framework + persistência** (`StageOutcome`, `Candidate`, escrita do JSON em `eval_results/_recommend_runs/`). Base de todas seguintes. ~3h.
7. **Etapas 3–6**: adapter sobre agentes existentes (memória D5 ligada). ~1h cada.
8. **Container Hermes do TradingAgents** (estudo separado): docker-compose, .env, config.yaml, bot Telegram. ~1h.
9. **Skill `cron/daily-recommend.md`** com prompt + delivery Telegram. ~30min.
10. **Backtest sintético**: rodar em 20 dias passados (subset XNYS), comparar P&L e custo com cascata "tudo em todos". Valida ROI da economia. ~3h.

Estimativa total: ~15–20h de implementação até primeira recomendação automática chegando no Telegram.

---

## Escopo deste estudo (e o que ficou de fora)

**Dentro:**
- Design da cascata eliminatória, ordem das etapas, gates skip-on-zero.
- Schema de contrato entre etapas, persistência, configuração.
- Esqueleto de implementação no Typer + módulo novo.

**Fora (estudos futuros):**
- Backtest framework: como medir se a cascata gera retorno? (estudo separado)
- Position sizing e portfolio constraints (limites por setor, beta, correlação).
- Aprendizado contínuo: agentes ficam melhores com runs anteriores via memória? Avaliar.
- Integração com broker (Alpaca / IB) para execução automatizada — alto risco, fora deste escopo.

---

## Apêndice — referências de arquivo

- Pipeline atual: [`tradingagents/graph/setup.py`](../tradingagents/graph/setup.py), [`graph/trading_graph.py`](../tradingagents/graph/trading_graph.py)
- Estados: [`agents/utils/agent_states.py`](../tradingagents/agents/utils/agent_states.py)
- Config: [`default_config.py`](../tradingagents/default_config.py)
- Screener pronto: [`tradingagents/screener/technical_screener.py`](../tradingagents/screener/technical_screener.py)
- CLI: [`cli/main.py`](../cli/main.py) (Typer; entrypoint `app()` na linha ~1196)
- Persistência atual: `eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json`
