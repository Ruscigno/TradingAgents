"""Persistent JSON file cache for bot results and configuration.

Stores to ``~/.tradingagents/cache/``:
  - ``last_analyze.json`` — last full analysis result
  - ``last_screen.json``  — last screener-only result
  - ``schedules.json``    — schedule definitions
  - ``config.json``       — runtime config (Telegram chat_id, etc.)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".tradingagents" / "cache"


class ResultsCache:
    """Read/write JSON files for bot results, schedules, and runtime config.

    All methods are safe to call concurrently from different threads — the
    filesystem serialises writes and the data volumes are tiny.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir or _CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Results ───────────────────────────────────────────────────────────────

    def save(self, command: str, result: dict[str, Any]) -> None:
        """Save a result dict for *command* (``'analyze'`` or ``'screen'``).

        Adds a ``saved_at`` timestamp automatically.
        """
        result = {**result, "saved_at": datetime.now(tz=timezone.utc).isoformat()}
        path = self._dir / f"last_{command}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        logger.debug(f"Saved {command} result to {path}")

    def load(self, command: str) -> dict[str, Any] | None:
        """Load the last saved result for *command*, or ``None`` if missing."""
        path = self._dir / f"last_{command}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load {path}: {exc}")
            return None

    # ── Schedules ─────────────────────────────────────────────────────────────

    def load_schedules(self) -> list[dict[str, Any]]:
        """Load all schedule definitions."""
        path = self._dir / "schedules.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load schedules: {exc}")
            return []

    def save_schedules(self, schedules: list[dict[str, Any]]) -> None:
        """Persist schedule definitions."""
        path = self._dir / "schedules.json"
        path.write_text(json.dumps(schedules, indent=2, default=str))

    # ── Runtime config (chat_id, etc.) ────────────────────────────────────────

    def save_chat_id(self, chat_id: int) -> None:
        """Persist the Telegram chat ID for push notifications."""
        config = self._load_config()
        config["chat_id"] = chat_id
        self._save_config(config)
        logger.info(f"Saved Telegram chat_id={chat_id}")

    def load_chat_id(self) -> int | None:
        """Return the stored Telegram chat ID, or ``None``."""
        return self._load_config().get("chat_id")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_config(self) -> dict[str, Any]:
        path = self._dir / "config.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_config(self, config: dict[str, Any]) -> None:
        path = self._dir / "config.json"
        path.write_text(json.dumps(config, indent=2, default=str))
