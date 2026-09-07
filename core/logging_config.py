from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PREFIX = "hesm"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | "
    "%(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class DailyFileHandler(logging.FileHandler):
    """Write directly to a date-named file and switch files at midnight."""

    def __init__(
        self,
        log_dir: Path,
        *,
        prefix: str = LOG_FILE_PREFIX,
        backup_count: int = 30,
        encoding: str = "utf-8",
    ) -> None:
        self.log_dir = log_dir.resolve()
        self.prefix = prefix
        self.backup_count = backup_count
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        super().__init__(self._path_for_date(self.current_date), encoding=encoding)
        self._cleanup_old_logs()

    def emit(self, record: logging.LogRecord) -> None:
        record_date = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
        if record_date != self.current_date:
            self._switch_to_date(record_date)
        super().emit(record)

    def _path_for_date(self, date_text: str) -> Path:
        return self.log_dir / f"{self.prefix}-{date_text}.log"

    def _switch_to_date(self, date_text: str) -> None:
        if self.stream:
            self.stream.close()
        self.current_date = date_text
        self.baseFilename = str(self._path_for_date(date_text).resolve())
        self.stream = self._open()
        self._cleanup_old_logs()

    def _cleanup_old_logs(self) -> None:
        if self.backup_count <= 0:
            return
        log_files = sorted(self.log_dir.glob(f"{self.prefix}-????-??-??.log"))
        for expired_file in log_files[:-self.backup_count]:
            try:
                expired_file.unlink(missing_ok=True)
            except OSError:
                continue


def _dated_log_file(log_dir: Path) -> Path:
    return log_dir / f"{LOG_FILE_PREFIX}-{datetime.now():%Y-%m-%d}.log"


def configure_logging(
    *,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> Path:
    """Configure HESM logging with daily file rotation."""
    target_dir = Path(log_dir).expanduser() if log_dir else DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = _dated_log_file(target_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler_exists = any(
        isinstance(handler, DailyFileHandler)
        and handler.log_dir == target_dir.resolve()
        for handler in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = DailyFileHandler(
            target_dir,
            backup_count=30,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if console and not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info("HESM logging initialized file=%s", log_file)
    return log_file


__all__ = ["DEFAULT_LOG_DIR", "DailyFileHandler", "configure_logging"]
