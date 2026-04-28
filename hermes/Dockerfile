# Hermes Agent dedicado ao TradingAgents.
# Roda gateway Telegram + cron diário (recommend) + executa o módulo
# tradingagents.recommend dentro do container.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Deps de sistema mínimas. Sem Playwright (gateway Telegram não usa).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ripgrep ffmpeg build-essential python3-dev libffi-dev procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 10000 -m -d /home/hermes hermes
RUN pip install --no-cache-dir "uv>=0.4"

# ---------- Hermes ----------
ARG HERMES_VERSION=v2026.4.23
RUN git clone --depth 1 --branch ${HERMES_VERSION} \
        https://github.com/NousResearch/hermes-agent.git /opt/hermes && \
    chown -R hermes:hermes /opt/hermes

USER hermes
WORKDIR /opt/hermes
RUN uv venv /opt/hermes/.venv && \
    . /opt/hermes/.venv/bin/activate && \
    uv pip install -e ".[all]"

# ---------- TradingAgents ----------
# Mounted at /workspace at runtime via docker-compose; install in editable mode.
USER root
RUN mkdir -p /workspace && chown hermes:hermes /workspace
USER hermes
ENV PATH="/opt/hermes/.venv/bin:${PATH}"
ENV HERMES_HOME=/home/hermes/.hermes
ENV PYTHONPATH=/workspace

WORKDIR /home/hermes
RUN hermes version

ENTRYPOINT ["hermes"]
CMD ["gateway", "run"]
