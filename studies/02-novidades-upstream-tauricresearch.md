# 02 — Relatório: Novidades do upstream TradingAgents (TauricResearch/main)

> **Status:** relatório de análise + plano de update
> **Autor:** Claude (sob direção de Sander)
> **Data:** 2026-04-27
> **Escopo:** comparação entre `yahoo-mds` (HEAD local), `origin/main` (fork Ruscigno), e `main` do upstream `TauricResearch/TradingAgents`.

---

## TL;DR

- O **fork local** (`yahoo-mds`) está **30 commits atrás** do upstream original. Divergiu em **`589b351` (v0.2.2)**.
- Você tem **4 commits locais** importantes em cima dessa base — todos dignos de manter.
- **3 mudanças upstream impactam diretamente o estudo da cascata** (estudo `01-...`) e podem nos poupar trabalho de implementação:
  1. **Structured-output Trader / Research Manager / Portfolio Manager** → exatamente o que pedimos no D2 (PASS/NO_PASS em JSON).
  2. **Persistent append-only decision log** substituindo memória BM25 → casa com a decisão D5 (memória ligada).
  3. **LangGraph checkpoint resume** → robustez para a cascata (uma run de 50 tickers que crashou na etapa 5 não precisa começar do zero).
- **Risco de update**: alto em **3 arquivos** (`default_config.py`, `Dockerfile`, `pyproject.toml`), médio em mais 2. A maioria dos seus 4 commits locais é em arquivos novos, então *cherry-pick* dirigido funciona melhor que rebase em massa.
- **Recomendação:** **atualizar em duas fases.** Fase A (segura, ~1h): pegar fixes pontuais e features que não conflitam. Fase B (mais cuidadosa, antes de implementar a cascata): adotar structured-outputs + decision log para construir a cascata em cima do código moderno.

---

## Topologia atual

```
                              upstream/main (TauricResearch)
                              ├── 7c37249  v0.2.4 release
                              ├── 4016fd4  fix base_url leak
                              ├── bba1477  ★ structured-output Trader/Research
                              ├── 0fda245  ★ structured-output Portfolio + 5-tier rating
                              ├── 4cbd4b0  ★ LangGraph checkpoint resume
                              ├── ebd2e12  ★ persistent decision log (replaces BM25)
                              ├── 8536cca  ignore CLAUDE.md
                              └── 872b063  utf-8 explicit encoding
                                       │  (6 commits — não estão no seu fork ainda)
                                       │
                              origin/main (Ruscigno) ← fa4d01c
                              ├── 24 commits desde v0.2.2 ──┐
                              │                              │
─────────── 589b351 (v0.2.2) ──┴──────────────────────────────┴─── ANCESTRAL
                                       │
                                       │ (sua branch yahoo-mds)
                                       ├── 469b2c5  MDS data vendor + screener integration
                                       ├── bb5ae1c  YAML per-ticker thresholds
                                       ├── ab61449  bot infra (MCP, scheduler, Telegram)
                                       └── cd3bc4b  local TCP proxy  ← HEAD
```

**Comandos pra reproduzir:**
```bash
git log --oneline HEAD..origin/main           # 24 commits faltando
git log --oneline origin/main..HEAD           # 4 commits seus
git rev-list --count HEAD..origin/main        # = 24
```

**Origin atualizado em** Apr 13/2026, **upstream** em Apr 25/2026 — origin está ~12 dias atrás do upstream.

---

## ★ Mudanças upstream que impactam o estudo da cascata

### 1. Structured-output Trader / Research Manager / Portfolio Manager

**Commits:** `bba1477` (Trader + Research Manager), `0fda245` (Portfolio Manager + 5-tier rating).
**Status:** v0.2.4 (no upstream, NÃO no origin/main ainda).
**Por que importa:** o estudo `01` (decisão D2) define que cada etapa LLM emite `{verdict: PASS|NO_PASS, confidence: 0..1}` em JSON estruturado. O upstream **acabou de fazer exatamente isso** para os 3 agents de decisão (etapas 5 e 6 da nossa cascata).

**Implicação prática:**
- Em vez de escrever do zero o "wrapper de structured output" ao redor desses agents, **adotar o que upstream já tem** poupa horas e fica alinhado com a comunidade.
- Os schemas deles podem não ser idênticos aos nossos (eles têm "5-tier rating" — provavelmente `Strong Buy / Buy / Hold / Sell / Strong Sell`). Precisamos mapear `verdict + confidence` em cima ou abraçar o schema de 5 tiers.

**Risco:** muda o shape do `AgentState` ([`tradingagents/agents/utils/agent_states.py`](../tradingagents/agents/utils/agent_states.py)) → o commit `ab61449` (bot infrastructure) consome `AgentState` em [`tradingagents/bot/results_cache.py`](../tradingagents/bot/results_cache.py) e [`mcp_server.py`](../tradingagents/bot/mcp_server.py) — pode quebrar.

### 2. Persistent append-only decision log (substitui memória BM25)

**Commits:** `ebd2e12`, `6abc768`.
**Status:** v0.2.4 (no upstream, NÃO no origin/main).
**O que muda:** a memória de cada agente, que era um índice BM25 vetorial em [`tradingagents/agents/utils/memory.py`](../tradingagents/agents/utils/memory.py), passou a ser um **log persistente append-only de decisões anteriores**.

**Por que importa:**
- Estudo `01` D5: queremos memória ligada por default. O log persistente é estruturalmente melhor que BM25 para nosso caso:
  - Reproduzível bit-a-bit (sem embeddings).
  - Auditável (legível à mão).
  - Não precisa de retriever (que pode trazer ruído).
- Combina perfeitamente com o JSON `eval_results/_recommend_runs/` que já planejamos.

**Risco:** **API breaking** para quem usa memory. Verificar se o commit `ab61449` (bot) acessa memory diretamente. Provavelmente não, mas confirmar.

### 3. LangGraph checkpoint resume

**Commit:** `4cbd4b0`.
**Status:** v0.2.4 (upstream apenas).
**O que faz:** salva checkpoints do graph LangGraph em disco; se o processo crashar (LLM timeout, OOM, rate-limit), retoma do último checkpoint.

**Por que importa:** cascata processa muitos tickers (50→1). Na etapa 5/6 (deep analysis), uma falha transiente em 1 ticker no batch de 3 hoje volta tudo ao zero. Com checkpoint, retoma só o ticker que falhou.

**Risco:** mínimo — é additive na infra de execução.

### 4. Dynamic OpenRouter model selection

**Commit:** `4f965bf` (#482, #337).
**Status:** já em origin/main.
**O que faz:** CLI procura modelos disponíveis no OpenRouter dinamicamente (em vez de lista hardcoded).
**Por que importa:** usamos OpenRouter (Kimi K2.6). UX melhor para futuros experimentos com outros modelos. Não-breaking.

---

## Outras mudanças relevantes (não-críticas mas úteis)

| Commit | Categoria | Status | Impacto |
|---|---|---|---|
| `b0f6058` | feat: providers DeepSeek/Qwen/GLM/Azure | origin | Nenhum direto (usamos Kimi); mas refatorou `llm_clients/factory.py` — pode tocar code path. |
| `10c136f` | feat: Docker support | origin | **Conflita** com seu `Dockerfile` em `ab61449`. Ver seção de riscos. |
| `e75d17b` | chore: defaults para GPT-5.4 | origin | Nenhum direto (usamos Kimi). Mas muda `default_config.py` — pode resolver bem com 3-way merge. |
| `6cddd26` | feat: multi-language para reports | origin | Permite gerar reports em PT-BR. Nice-to-have. |
| `e111388` | fix: previne look-ahead bias em backtest | origin | **Crítico para o passo 10 do estudo `01`** (backtest sintético). Adoptar. |
| `58e9942` | fix: base_url para Google/Anthropic clients | origin | Não nos afeta (usamos OpenRouter). |
| `4016fd4` | fix: stop leaking OpenAI base_url | upstream | Bom de pegar; afeta provider routing. |
| `872b063` | fix: utf-8 explicit encoding | upstream | Cosmético em Mac/Linux, importante em Windows. |

---

## Bug fixes que valem cherry-pick imediato

Independente da estratégia geral de update, esses são fixes pequenos, não-breaking, e fazem o código local ficar melhor:

| Commit | Fix | Por que pegar |
|---|---|---|
| `28d5cc6` | missing `import pandas` em `y_finance.py` | bug óbvio, qualquer hora pode te morder |
| `7269f87` | Portfolio Manager agora lê trader proposal + research plan | era um **bug semântico** — o portfolio manager estava ignorando contexto upstream nas decisões. **Importante.** |
| `78fb66a` | normalize indicator names to lowercase | melhora robustez do screener (afeta seu commit `bb5ae1c`) |
| `7004dfe` | remove hardcoded Google endpoint que dava 404 | irrelevante (não usamos Google), mas zero custo de pegar |
| `ae8c8ae` | gracefully handle invalid indicator names | robustez — afeta seu screener |
| `f3f58bd` | yf_retry em yfinance news fetchers | resiliência contra rate-limit do Yahoo (relevante já que `mds_client.py` é via Yahoo) |
| `fa4d01c` | tool call logging + memory score normalization | dois bugs sutis em tool-calling — vale pegar antes de implementar a cascata |

---

## Breaking changes e dependency bumps

| Commit | O que muda | Risco |
|---|---|---|
| `bdc5fc6` | bump `langchain-google-genai` ≥ 4.0.0 | Não usamos Google, mas o `pyproject.toml` cascata disso pode pegar bibliotecas relacionadas. **Médio.** |
| `bdb9c29` | refactor: imports + configurable results path | Refator interno. Pode mover símbolos importados pelo seu código. **Médio.** |
| `b0f6058` | factory de LLM clients refatorada | Adicionou DeepSeek/Qwen/GLM/Azure → mudou shape do `llm_clients/factory.py`. Seu `cd3bc4b` (TCP proxy) é em arquivo separado, não conflita. **Baixo-médio.** |
| `ebd2e12` | substitui BM25 memory por decision log | **API muda.** Se algum código local importa `agents.utils.memory` ainda, quebra. Verificar antes. **Médio-alto se houver uso.** |
| `bba1477` / `0fda245` | structured-output em 3 managers | **`AgentState` ganha campos novos.** Seu `bot/results_cache.py` consome esse estado — pode dessincronizar. **Médio.** |
| `10c136f` | upstream agora tem Dockerfile próprio | Você já tem **seu** Dockerfile (do commit `ab61449`). **Alto.** Decidir: ficar com o seu, ou usar o deles e portar suas adições. |

---

## Riscos por commit local (seus 4 commits)

### `cd3bc4b` — TCP proxy (1 arquivo novo, isolado)
- Toca apenas `local_network_proxy.py` — arquivo novo, sem precedente upstream.
- **Risco de merge: zero.** Sobrevive a qualquer rebase trivialmente.

### `bb5ae1c` — YAML thresholds para o screener técnico
- Toca `tradingagents/screener/technical_screener.py` (compartilhado com upstream `78fb66a`) e `screener/config_loader.py` (novo).
- **Risco: baixo-médio.** Conflito provável em `technical_screener.py` por causa de `78fb66a`, mas pequeno (só normalização de nome de indicador).
- O `screener.yaml` versionado é seu — upstream não tem.

### `469b2c5` — MDS como data vendor + integration
- Toca arquivos novos (`dataflows/mds_*.py`, `screener/__init__.py`, `screen_and_trade.py`, docs `.claude/*`) — **sem conflito**.
- Toca `dataflows/interface.py`, `default_config.py`, `screener/technical_screener.py` — **conflitos prováveis**:
  - `default_config.py` mudou em `e75d17b` (defaults GPT-5.4) e `bdb9c29` (configurable results path). 3-way merge resolverá.
  - `technical_screener.py` é compartilhado com `bb5ae1c`.
  - `dataflows/interface.py` pode ter sido mexido junto com `bdb9c29` (refactor). Verificar.
- **Risco: médio.**

### `ab61449` — Bot infrastructure (MCP, scheduler, Telegram)
- Toca `tradingagents/bot/*` — todos arquivos novos, **sem conflito**.
- Toca `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `main.py` — **conflitos prováveis**:
  - **Dockerfile + docker-compose.yml + .dockerignore** colidem com o `10c136f` upstream (que adicionou Docker oficial). Decisão: manter o seu (que tem MCP/scheduler) ou portar.
  - `pyproject.toml` colide com `bdc5fc6` (langchain-google-genai bump). Resolver dependency bump conjunto.
  - `default_config.py` (segundo commit a tocar) — mais um turno de 3-way merge.
- **Risco: alto** — esse commit é o mais "invasivo" no projeto e é onde a maioria dos atritos vai estar.

---

## Estratégia recomendada de update

### Fase A — Cherry-picks seguros (~1h)

Cherry-pick dos commits **isolados** e **úteis** sem rebasear. Não toca nos seus 4 commits — só adiciona em cima.

**Ordem sugerida** (do mais seguro pro menos):

1. `28d5cc6` (pandas import fix) — 1 linha
2. `7004dfe` (remove hardcoded Google endpoint) — irrelevante mas trivial
3. `f3f58bd` (yf_retry em yfinance) — resiliência
4. `ae8c8ae` (graceful invalid indicator) — robustez
5. `78fb66a` (lowercase indicators) — provável conflito leve com `bb5ae1c`, resolvível
6. `7269f87` (Portfolio Manager lê context) — bug semântico importante, sem conflito esperado
7. `e111388` (look-ahead bias fix) — vital pro backtest
8. `fa4d01c` (tool call logging + memory) — antes de implementar cascata

Ao final dessa fase, você está ~30% mais perto de origin/main, com zero risco para as features locais.

### Fase B — Rebase total (antes da implementação da cascata)

Antes de implementar o estudo `01`, rebase `yahoo-mds` em `upstream/main` (= origin/main + os 6 do upstream):

- Resolve todos os conflitos de uma vez.
- Aceita o `Dockerfile` upstream OU mantém o seu — escolha consciente.
- Migra `agents/utils/memory.py` → decision log.
- Adota structured outputs nos managers.
- Custo: ~3–5h, alguns conflitos manuais.
- Benefício: você fica sincronizado e ganha checkpoint resume + structured outputs nativos.

### Fase C — Implementar o estudo `01`

Já em cima do código moderno. Reusa structured outputs upstream em vez de implementar do zero. Reusa decision log em vez do BM25 antigo.

---

## Plano de execução — Fase A (cherry-picks)

### Preparação

```bash
cd /Users/sander/projects/TradingAgents

# 1) Snapshot da branch atual (segurança)
git branch yahoo-mds-snapshot-pre-update

# 2) Adicionar upstream como remote
git remote add upstream https://github.com/TauricResearch/TradingAgents.git
git fetch upstream

# 3) Confirmar estado de origem dos commits
git log --oneline upstream/main | head -5    # confirma os 6 commits "pós-fork"
git log --oneline origin/main | head -5      # confirma os 24 commits "no Ruscigno"
```

### Cherry-picks

| # | Commit | Subject | Risco | Comando |
|---|---|---|---|---|
| 1 | `28d5cc6` | missing pandas import in y_finance.py | zero | `git cherry-pick -x 28d5cc6` |
| 2 | `7004dfe` | remove hardcoded Google endpoint | zero | `git cherry-pick -x 7004dfe` |
| 3 | `f3f58bd` | yf_retry em yfinance news fetchers | baixo | `git cherry-pick -x f3f58bd` |
| 4 | `e111388` | prevent look-ahead bias em backtest | baixo | `git cherry-pick -x e111388` |
| 5 | `7269f87` | portfolio manager reads context | baixo | `git cherry-pick -x 7269f87` |
| 6 | `ae8c8ae` | graceful invalid indicator names | baixo-médio | `git cherry-pick -x ae8c8ae` |
| 7 | `78fb66a` | lowercase indicators normalization | médio (conflito provável) | `git cherry-pick -x 78fb66a` |
| 8 | `fa4d01c` | tool call logging + memory normalization | baixo-médio | `git cherry-pick -x fa4d01c` |

### Resolução de conflitos esperados

**Provável: #7 (`78fb66a`)** vai conflitar em `technical_screener.py`. Estratégia:
- Aceitar a mudança upstream (lowercase) e adaptar nosso config loader (`bb5ae1c`) pra também normalizar os keys YAML em lowercase ao carregar.
- Editar conflito → `git add` → `git cherry-pick --continue`.

**Possível: #6 (`ae8c8ae`)** pode conflitar em `dataflows/`. Aceitar mudança upstream (gracefully handle) é compatível com o nosso routing por vendor.

### Verificação após cada cherry-pick

```bash
python -c "import tradingagents; print('import ok')"
pytest tests/ -x --no-cov -q 2>&1 | tail -5
```

Se algum teste quebrar inesperadamente (não relacionado ao commit acabado de aplicar), parar e investigar.

### Ao final da Fase A

- 8 commits de fixes upstream aplicados em cima dos 4 seus.
- Histórico ainda é linear (cherry-picks, não merges).
- Próximo passo: Fase B antes de iniciar a implementação do estudo `01` (cascata).

---

## Checklist antes de começar Fase B

- [ ] Backup da branch atual: `git branch yahoo-mds-snapshot-pre-rebase`
- [ ] Verificar se algum código em `tradingagents/bot/` ou `screener/` importa `tradingagents.agents.utils.memory` — se sim, planejar migração
- [ ] Comparar `Dockerfile` local vs upstream antes de decidir qual ficar
- [ ] Rodar `pip-tools` ou `uv lock --upgrade` após resolver `pyproject.toml`

---

## Decisão sobre Pull Request para upstream

**Não criar agora.** Reavaliar após Fase B + cascata implementada — quando teremos:
- Fork próximo de upstream (rebase concluído).
- Uso real validando o que sobreviveu (cascata rodando em produção dá feedback).
- Feeling melhor das convenções deles.

Análise dos 4 commits locais como candidatos a PR:
- `bb5ae1c` (YAML thresholds): self-contained, mas screener não é parte central do projeto upstream — aceite incerto.
- `ab61449` (bot infra): opinionado em deps; maintainers podem preferir design próprio.
- `469b2c5` (data vendor): provavelmente colide com refatorações internas.
- `cd3bc4b` (TCP proxy): muito local — não é PR material.

Melhor não criar do que ter baixa expectativa de merge.

---

## Apêndice — referências de arquivos do upstream a ler antes da Fase B

- Decision log replacement: `tradingagents/agents/utils/memory.py` na branch upstream/main (vs versão BM25 da v0.2.2).
- Structured-output Trader: o commit `bba1477` na árvore `tradingagents/agents/trader/`.
- Structured-output Portfolio Manager: `0fda245` em `tradingagents/agents/managers/`.
- Checkpoint resume: `4cbd4b0` em `tradingagents/graph/`.
- Dockerfile oficial: `10c136f`.
- Dependency snapshot novo: `pyproject.toml` em upstream/main.
