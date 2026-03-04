"""Unified logging utility for Safety View Agent."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to store log files (default: "logs")
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    use_file_logging = True
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: Cannot create log directory '{log_dir}': {e}", file=sys.stderr)
        use_file_logging = False

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (rotating) - only if directory creation succeeded
    if use_file_logging:
        try:
            log_file = log_path / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB per file, 5 backups
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            print(f"Warning: Cannot create file handler: {e}", file=sys.stderr)

    # Console handler (for immediate feedback)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)
