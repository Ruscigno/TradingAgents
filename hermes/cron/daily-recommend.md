---
name: daily-recommend
schedule: "30 16 * * 1-5"
timezone: America/New_York
description: |
  Roda a cascata eliminatória de recomendações de trade após o close
  do mercado americano. Apenas em dias úteis (XNYS). Output é entregue
  no canal home do Telegram.
---

[SYSTEM: você está rodando como cron job. DELIVERY: a sua resposta final será
enviada ao canal home do Telegram automaticamente. Mantenha a resposta concisa
e em português.]

Tarefa diária — gerar recomendações de trade.

1. Use a tool `shell` para verificar se hoje é um dia útil do NYSE:

   ```
   cd /workspace && python -c "from tradingagents.recommend.calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 1)"
   ```

   Se o exit code for não-zero, responda apenas:
   "📅 Mercado fechado hoje — nenhuma recomendação será gerada."
   e encerre.

2. Se for dia útil, dispare a cascata completa:

   ```
   cd /workspace && python -m cli.main recommend --full --date $(date +%Y-%m-%d)
   ```

   Isso pode levar alguns minutos. Aguarde a saída completa.

3. O comando imprime no stdout (e persiste em
   `eval_results/_recommend_runs/{date}_{run_id}.json`):

   - resumo por etapa (in/passed/eliminated por etapa)
   - tabela final com tickers e ações (BUY/SELL)

4. Formate a resposta em markdown para o Telegram:

   ```
   🔔 *Recomendações de trade — {data}*
   Universo: {universe_size} → {N final} candidatas

   | # | Ticker | Action | Score | Stop | Target |
   |---|--------|--------|------:|-----:|-------:|
   ...

   Position size: 2% por trade (config)
   Custo do run: ${total_cost}
   Tempo: {duration}s
   Detalhes: eval_results/_recommend_runs/{date}_*.json
   ```

5. Se zero candidatas chegaram à etapa final, responda:
   "ℹ️ Cascata rodou ({duration}s). Nenhuma candidata sobreviveu até a etapa
   final. Etapa de saída: {última etapa que rodou}. Custo: ${total_cost}."

Envie a resposta ao canal home.
