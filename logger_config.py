import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler("system.log", maxBytes=5_000_000, backupCount=5)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s : %(message)s'
    )

    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger