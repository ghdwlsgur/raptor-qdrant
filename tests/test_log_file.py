import logging

from loguru import logger as loguru_logger

from raptor_qdrant.core import logger as logging_setup
from raptor_qdrant.core.config import settings


def test_file_sink_receives_records_from_stdlib_logging(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "raptor.log"
    monkeypatch.setattr(settings, "LOG_FILE", str(target))

    logging_setup.configure_logging()
    logging.getLogger("raptor_qdrant.test").info("파일로도 남는 진행 로그")
    loguru_logger.complete()

    assert "파일로도 남는 진행 로그" in target.read_text(encoding="utf-8")


def test_empty_log_file_setting_disables_the_sink(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOG_FILE", "  ")

    assert logging_setup.add_file_sink("INFO") is None
    assert list(tmp_path.iterdir()) == []
