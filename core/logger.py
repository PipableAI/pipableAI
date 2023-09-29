import logging
import logging.config
import os
from pathlib import Path

# Define a constant for the log file path
module_directory = Path(__file__).resolve().parent
DEV_LOG_FILE_PATH = os.path.join(module_directory, "../logs/dev.log")

# Create the "logs" folder if it doesn't exist
logs_directory = os.path.join(module_directory, "../logs")
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)


def dev_logger(log_file_path=DEV_LOG_FILE_PATH):
    # Step 1: Create a logger
    logger = logging.getLogger("_dev_logger")
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)

    # Step 2: Create a custom handler for writing to the log file
    file_handler = logging.FileHandler(log_file_path)

    # Step 3: Create a custom handler for displaying logs on the console
    console_handler = logging.StreamHandler()

    # Step 4: Create a custom log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s: %(funcName)s : line %(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 5: Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
