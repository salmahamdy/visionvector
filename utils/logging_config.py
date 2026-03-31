"""Centralised logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_dir: str | Path = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(log_dir) / "app.log"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=log_format, handlers=handlers)
