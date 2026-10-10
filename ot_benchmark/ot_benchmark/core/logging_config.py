"""Structured logging configuration for the OT Benchmark system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s"
DATE_FORMAT = "%H:%M:%S"
LOG_FORMAT_COLOR = (
    "\033[2m%(asctime)s\033[0m "
    "%(levelname_color)s[%(levelname)-7s]\033[0m "
    "\033[1m%(name)s\033[0m — %(message)s"
)

LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in LEVEL_COLORS:
            record.levelname_color = LEVEL_COLORS[levelname]
        else:
            record.levelname_color = ""
        return super().format(record)


def _steamlit_warning_filter() -> None:
    """Suppress noisy file-watcher warnings from Streamlit / watchdog."""
    try:
        import warnings
        import logging as _logging
        warnings.filterwarnings("ignore", message=".*was deleted but is still being monitored.*")
        _logging.getLogger("watchdog.observers.inotify_buffer").setLevel(_logging.ERROR)
    except Exception:  # best-effort; never fail startup on log suppression
        pass


_LOGGER_CACHE: dict[str, logging.Logger] = {}
_SETUP_DONE: bool = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
    json_format: bool = False,
) -> None:
    """Configure root logger for the benchmark system.

    Args:
        level: DEBUG, INFO, WARNING, or ERROR.
        log_file: If provided, also write rotated logs to this file.
        json_format: If True, emit JSON-structured logs.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    _steamlit_warning_filter()

    root = logging.getLogger("ot_benchmark")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Console handler with colors
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    if json_format:
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    else:
        console.setFormatter(ColoredFormatter(LOG_FORMAT_COLOR, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module, namespaced under ot_benchmark.

    Args:
        name: Module name (e.g., 'benchmark_engine', 'data_generator').

    Returns:
        Logger instance.
    """
    full_name = f"ot_benchmark.{name}"
    if full_name not in _LOGGER_CACHE:
        _LOGGER_CACHE[full_name] = logging.getLogger(full_name)
    return _LOGGER_CACHE[full_name]
