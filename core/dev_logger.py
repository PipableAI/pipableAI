import logging
import logging.config
import os
from pathlib import Path


def dev_logger(log_file_path: str = None) -> logging.Logger:
    """
    Configures and returns a logger for development purposes.

    Args:
        log_file_path (str): Optional. Path to the log file. If not provided, logs will only be displayed on the console.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Step 1: Create a logger
    logger = logging.getLogger("_dev_logger")
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s: %(funcName)s : line %(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # Step 2: Log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Step 3: Create a custom handler for writing to the log file or console
    if log_file_path:
        # Log to file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
