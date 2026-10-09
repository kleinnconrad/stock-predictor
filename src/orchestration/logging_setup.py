import logging
import os
import sys
from typing import Optional


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configures root logging for command line entry points.

    Args:
        log_file (Optional[str]): Additional file to write the log to.
        level (int): Minimum log level.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
