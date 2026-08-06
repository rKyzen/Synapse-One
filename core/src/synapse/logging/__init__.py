"""Structured logging subsystem.

Every subsystem logs through :func:`synapse.logging.get_logger` with a
hierarchical name (``synapse.providers.ollama``). Structured JSON output is
available in production; human-readable console is default for development.
Bind context at call sites; never use bare ``print``.
"""

from __future__ import annotations

import logging
import sys

import structlog

from synapse.config.settings import Settings


def get_logger(name: str = "synapse") -> structlog.stdlib.BoundLogger:
    """Return a structured logger for a subsystem, e.g. ``synapse.hardware``."""
    return structlog.get_logger(name)


def configure(settings: Settings) -> None:
    """Configure the global structlog pipeline from application settings.

    Call exactly once from the composition root.
    """
    level = settings.logging.level.upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
    )

    for logger_name, override in settings.logging.overrides.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, override.upper(), logging.WARNING))

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if settings.logging.json_lines:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Silence noisy third-party loggers by default.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
