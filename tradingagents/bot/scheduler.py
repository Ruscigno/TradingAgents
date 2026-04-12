"""APScheduler-based background scheduler for recurring analysis runs.

Scheduled jobs run ``run_analysis()`` and push results to Telegram via the
Bot API directly (``POST /bot{token}/sendMessage``).

Scheduler state (schedule definitions) is persisted to
``~/.tradingagents/cache/schedules.json`` so schedules survive process
restarts.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .analysis_tools import run_analysis
from .results_cache import ResultsCache

logger = logging.getLogger(__name__)

_cache = ResultsCache()


class TradingScheduler:
    """Manages recurring analysis jobs with APScheduler.

    Each schedule stores its own parameters independently — changing
    ``screener.yaml`` or CLI defaults does **not** affect existing schedules.
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(daemon=True)
        self._schedules: dict[str, dict[str, Any]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler and restore persisted schedules."""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("TradingScheduler started")
        self._restore_schedules()

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("TradingScheduler stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def create_schedule(
        self,
        recurrence_hours: int,
        start_datetime: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Create a recurring analysis schedule.

        Args:
            recurrence_hours: How often to run (e.g. 24 for daily).
            start_datetime:   First run time as ``YYYY-MM-DDTHH:MM``
                              (default: now + recurrence_hours).
            params:           Analysis parameters forwarded to ``run_analysis()``.

        Returns:
            A schedule ID string (e.g. ``sched_001``).
        """
        schedule_id = self._next_id()
        params = params or {}

        if start_datetime:
            first_run = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        else:
            first_run = datetime.now(tz=timezone.utc) + timedelta(hours=recurrence_hours)

        trigger = IntervalTrigger(
            hours=recurrence_hours,
            start_date=first_run,
        )

        self._scheduler.add_job(
            self._run_scheduled_job,
            trigger=trigger,
            id=schedule_id,
            args=[schedule_id, params],
            replace_existing=True,
        )

        schedule_def = {
            "id": schedule_id,
            "recurrence_hours": recurrence_hours,
            "start_datetime": first_run.isoformat(),
            "params": params,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._schedules[schedule_id] = schedule_def
        self._persist_schedules()

        logger.info(f"Created schedule {schedule_id}: every {recurrence_hours}h starting {first_run}")
        return schedule_id

    def list_schedules(self) -> list[dict[str, Any]]:
        """Return all active schedule definitions."""
        result = []
        for sid, sdef in self._schedules.items():
            job = self._scheduler.get_job(sid)
            next_run = str(job.next_run_time) if job and job.next_run_time else "unknown"
            result.append({
                **sdef,
                "next_run": next_run,
            })
        return result

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule by ID. Returns ``True`` if it existed."""
        if schedule_id not in self._schedules:
            return False

        try:
            self._scheduler.remove_job(schedule_id)
        except Exception:
            pass  # Job may already have been removed

        del self._schedules[schedule_id]
        self._persist_schedules()
        logger.info(f"Deleted schedule {schedule_id}")
        return True

    # ── Job execution ─────────────────────────────────────────────────────────

    def _run_scheduled_job(self, schedule_id: str, params: dict[str, Any]) -> None:
        """Execute a scheduled analysis and push results to Telegram."""
        logger.info(f"Running scheduled job {schedule_id}")

        try:
            result = run_analysis(**params)
        except Exception as exc:
            logger.error(f"Scheduled job {schedule_id} failed: {exc}")
            result = {"error": str(exc), "schedule_id": schedule_id}

        # Push to Telegram
        self._push_to_telegram(schedule_id, result)

    def _push_to_telegram(self, schedule_id: str, result: dict[str, Any]) -> None:
        """Send results to Telegram via the Bot API.

        Reads the bot token from ``TELEGRAM_BOT_TOKEN`` env var and the
        chat_id from the results cache.
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = _cache.load_chat_id()

        if not token:
            logger.warning("TELEGRAM_BOT_TOKEN not set — cannot push scheduled results")
            return
        if not chat_id:
            logger.warning(
                "No Telegram chat_id stored — send /start to the bot first. "
                "Skipping push for schedule %s",
                schedule_id,
            )
            return

        # Format the message
        message = self._format_result_message(schedule_id, result)

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=15,
            )
            if not resp.ok:
                logger.error(f"Telegram sendMessage failed: {resp.status_code} {resp.text}")
        except Exception as exc:
            logger.error(f"Failed to push to Telegram: {exc}")

    @staticmethod
    def _format_result_message(schedule_id: str, result: dict[str, Any]) -> str:
        """Format an analysis result dict as a Telegram-friendly message."""
        if "error" in result:
            return f"⚠️ *Scheduled run {schedule_id} failed*\n\n`{result['error']}`"

        date = result.get("date", "unknown")
        decisions = result.get("trade_decisions", [])
        screening = result.get("screening", {})
        passed = screening.get("passed_count", 0)
        total = screening.get("total", 0)

        lines = [
            f"📊 *Scheduled Analysis — {date}*  (schedule: `{schedule_id}`)",
            f"Screener: {passed} / {total} passed",
            "",
        ]

        if not decisions:
            lines.append("No candidates to analyze.")
        else:
            for d in decisions:
                ticker = d.get("ticker", "?")
                signal = d.get("signal", "?")
                lines.append(f"• *{ticker}*: {signal}")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_schedules(self) -> None:
        _cache.save_schedules(list(self._schedules.values()))

    def _restore_schedules(self) -> None:
        """Reload persisted schedules and re-register them with APScheduler."""
        saved = _cache.load_schedules()
        for sdef in saved:
            sid = sdef["id"]
            try:
                trigger = IntervalTrigger(
                    hours=sdef["recurrence_hours"],
                    start_date=datetime.fromisoformat(sdef["start_datetime"]).replace(tzinfo=timezone.utc),
                )
                self._scheduler.add_job(
                    self._run_scheduled_job,
                    trigger=trigger,
                    id=sid,
                    args=[sid, sdef.get("params", {})],
                    replace_existing=True,
                )
                self._schedules[sid] = sdef
                logger.info(f"Restored schedule {sid}")
            except Exception as exc:
                logger.warning(f"Failed to restore schedule {sid}: {exc}")

    def _next_id(self) -> str:
        """Generate a short schedule ID like ``sched_001``."""
        existing = set(self._schedules.keys())
        for i in range(1, 10000):
            candidate = f"sched_{i:03d}"
            if candidate not in existing:
                return candidate
        return f"sched_{uuid.uuid4().hex[:6]}"
