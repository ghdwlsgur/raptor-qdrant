import logging
import sys
from pathlib import Path

from loguru import logger

from raptor_qdrant.core.config import settings


def kst_formatter(record):
    """한국 시간으로 포맷팅"""
    from datetime import timedelta, timezone

    # UTC+9 (한국 표준시)
    kst = timezone(timedelta(hours=9))
    record["time"] = record["time"].astimezone(kst)
    return record


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 원래 logging record의 위치 정보 사용
        logger.patch(
            lambda r: r.update(  # type: ignore[call-arg]
                name=record.name,
                module=record.module,
                function=record.funcName,
                line=record.lineno,
                file=record.pathname,
            )
        ).log(level, record.getMessage())


def configure_logging():
    # 모든 기존 핸들러 제거
    logging.root.handlers.clear()

    # 모든 기존 로거의 핸들러를 제거
    for name in logging.root.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        logger_obj.handlers.clear()
        logger_obj.propagate = True  # 부모 로거로 전파하도록 설정

    # 루트 로거에만 InterceptHandler 추가
    intercept_handler = InterceptHandler()
    logging.root.addHandler(intercept_handler)
    logging.root.setLevel(logging.DEBUG)

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
            filter=kst_formatter,
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
            filter=kst_formatter,
        )

    log_file = add_file_sink(log_level)

    logger.info(
        f"logging system is configured successfully: {environment}"
        + (f", log file {log_file}" if log_file else "")
    )


def add_file_sink(log_level: str) -> Path | None:
    """LOG_FILE 이 있으면 파일 sink 를 단다. 못 만들면 stdout 만으로 간다.

    몇 시간 도는 인덱싱의 진행을 다른 터미널에서 tail 로 보려면 파일이
    필요하다. 색 없이 평문으로 쓰고 10MB 마다 돌려 5개까지 둔다.
    """
    target = settings.LOG_FILE.strip()
    if not target:
        return None

    path = Path(target).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            filter=kst_formatter,
        )
    except OSError as e:
        logger.warning(f"cannot write log file {path}: {e}")
        return None
    return path
