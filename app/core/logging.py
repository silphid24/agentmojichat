"""Logging configuration"""

import logging
import sys
from typing import Optional

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        if not settings.debug:
            return super().format(record)

        log_color = self.COLORS.get(record.levelno)
        record.levelname = f"{log_color}{record.levelname}{self.reset}"
        return super().format(record)


def setup_logging(
    name: Optional[str] = None, level: Optional[int] = None
) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name or "moji")

    if logger.handlers:
        return logger

    log_level = level or (logging.DEBUG if settings.debug else logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    format_string = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
    formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = setup_logging()
