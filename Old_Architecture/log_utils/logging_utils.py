import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = "orchestration.log", level: int = logging.INFO) -> None:
    """Configure application wide logging.

    All messages are written both to the console and to a central log file.

    Parameters
    ----------
    log_file: str
        Path of the log file. A new file is created if it does not exist.
    level: int
        The logging level to configure the root logger with.
    """

    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate log entries if called multiple times
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)