import logging
from rich.logging import RichHandler

def setup_logging(level: int = logging.INFO) -> None:
    """Initialize standard framework logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a module-specific logger."""
    return logging.getLogger(name)
