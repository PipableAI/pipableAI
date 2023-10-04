import logging
import logging.config
import os
from pathlib import Path


def dev_logger(log_file_path: str = None) -> logging.Logger:
    """
    Configures and returns a logger for development purposes.

    Args:
        log_file_path (str, optional): Path to the log file.
            If provided, logs will be saved to this file. Defaults to None.

    Returns:
        logging.Logger: Configured logger object.

    .. code-block:: python

        # Example usage:
        logger = dev_logger("example.log")
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")
    """
    logger = logging.getLogger("_dev_logger")
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s: %(funcName)s : line %(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a custom handler for writing to the log file or console
    if log_file_path:
        # Log to file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
