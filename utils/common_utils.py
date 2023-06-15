import time
import logging

from pathlib import Path

fmt = "%(asctime)s - %(name)s - %(levelname)s -%(message)s"
fname = f"{time.strftime('%Y%m%d_%H%M%S',time.localtime())}.log"


# initial logger
def init_logger(name, log_file=None, log_file_level=logging.NOTSET):
    logger = logging.getLogger(name=name)

    logger.setLevel(logging.INFO)

    if isinstance(log_file, Path):
        log_file = str(log_file)

    formatter = logging.Formatter(
        fmt=fmt, datefmt="%m/%d/%y %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.handlers = [ch]

    if log_file and log_file != "":
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_file_level)
        logger.addHandler(fh)

    return logger
