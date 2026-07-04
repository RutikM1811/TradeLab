import sys
from pathlib import Path

from loguru import logger


def configure_logger(log_level: str = "INFO") -> None:
    """Configure Atlas application logging."""

    # Remove Loguru's default handler
    logger.remove()

    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    logger.add(
        log_dir / "atlas.log",
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        )