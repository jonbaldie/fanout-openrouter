import logging
import sys
from logging import Formatter, StreamHandler
from typing import Any

class JsonFormatter(Formatter):
    """
    Format log records as JSON for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include standard attributes that might be useful
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)

        # Include any extra attributes passed via the 'extra' kwarg
        if hasattr(record, "candidate_index"):
            log_entry["candidate_index"] = record.candidate_index
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        if hasattr(record, "error"):
            log_entry["error"] = record.error
        if hasattr(record, "attempt"):
            log_entry["attempt"] = record.attempt
        if hasattr(record, "max_attempts"):
            log_entry["max_attempts"] = record.max_attempts
        if hasattr(record, "delay"):
            log_entry["delay"] = record.delay
        if hasattr(record, "last_error"):
            log_entry["last_error"] = record.last_error
        if hasattr(record, "default_model"):
            log_entry["default_model"] = record.default_model

        import json
        return json.dumps(log_entry)

def configure_logging(level: int = logging.INFO, structured: bool = True) -> None:
    """
    Configure the root logger for the application.
    """
    root_logger = logging.getLogger()
    
    # Remove existing handlers to prevent duplicates if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(level)

    handler = StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        # Standard simple format for local dev
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    
    # Adjust overly verbose loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
