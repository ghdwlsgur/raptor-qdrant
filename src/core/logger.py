import logging
import sys
from types import FrameType
from typing import Optional
from loguru import logger
from src.core.config import settings


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: Optional[FrameType] = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in logging.root.manager.loggerDict:
        if name.startswith("uvicorn."):
            logging.getLogger(name).handlers = []

    logger.remove()

    log_level = settings.LOG_LEVEL.upper()
    environment = settings.ENVIRONMENT.lower()

    if environment == "production":
        logger.add(
            sys.stdout,
            level=log_level,
            format="{message}",
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            serialize=False,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"logging system is configured successfully: {environment}")
