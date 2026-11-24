"""Structured JSON logging for SignBridge.

Call :func:`configure_logging` once at process start (CLIs and the UI do
this), then ``log = get_logger(__name__)`` anywhere. Output is one JSON
object per line — machine-parseable, grep-friendly, and safe to ship to
any log aggregator.

``structlog`` is used when installed. Where it is not (for example a Kaggle
notebook that only imports the data/model code), a small stdlib fallback
with the same ``log.info("event", key=value)`` interface is used instead.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

structlog: Any
try:
    import structlog
except ImportError:  # pragma: no cover - exercised only without structlog
    structlog = None


class _FallbackLogger:
    """Minimal ``log.info("event", **fields)`` logger on top of stdlib logging."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        if self._log.isEnabledFor(level):
            payload = {"event": event, "level": logging.getLevelName(level).lower(), **fields}
            self._log.log(level, json.dumps(payload, default=str))

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure stdlib (+ structlog when available) for JSON output to stderr.

    Args:
        level: Minimum log level (stdlib constant, e.g. ``logging.DEBUG``).
    """
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a logger named after the calling module."""
    if structlog is None:
        return _FallbackLogger(name)
    return structlog.get_logger(name)
