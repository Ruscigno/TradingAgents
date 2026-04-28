"""LLM helpers for the recommend cascade.

The cascade uses Kimi K2.6 via OpenRouter (decision D0 in study 01) for all
LLM-driven stages. Loading config and instantiating the client lives here so
each stage doesn't repeat the boilerplate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from tradingagents.llm_clients.factory import create_llm_client


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Config used by every LLM-driven stage in the cascade."""

    provider: str = "openrouter"
    model: str = "moonshotai/kimi-k2.6"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2

    def assert_api_key_present(self) -> None:
        """Fail fast if the API key env var is unset.

        Catches the most common config error (forgot to set the key) before
        we kick off parallel calls and burn N×timeout seconds on auth errors.
        """
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                f"Required for LLM provider {self.provider!r}."
            )


def build_llm(config: LLMConfig | None = None) -> Any:
    """Return a configured LangChain-compatible LLM (chat) client.

    Uses :func:`tradingagents.llm_clients.factory.create_llm_client` so we
    inherit the project's existing OpenAI-compatible adapter, validators,
    and ``with_structured_output`` plumbing.
    """
    cfg = config or LLMConfig()
    cfg.assert_api_key_present()
    client = create_llm_client(
        provider=cfg.provider,
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    log.debug(
        "build_llm: provider=%s model=%s base_url=%s",
        cfg.provider, cfg.model, cfg.base_url,
    )
    return client.get_llm()
