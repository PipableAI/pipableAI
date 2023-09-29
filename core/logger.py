import logging

def PIPABLE_LOGGER_CREATE(logger_name, log_level="DEBUG"):
    '''
    Usage:
        PIPABLE_LOGGER = PIPABLE_LOGGER_CREATE(logger_name="PIPABLE_MAIN", log_level="INFO")
        PIPABLE_LOGGER.info(f"Pipable class instantiated from {path}")
        PIPABLE_LOGGER.critical(f"This is a critical log. I also support .error and .info")
    '''
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = levels.get(log_level, logging.DEBUG)
    log = logging.getLogger(logger_name)
    log.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)
    return log