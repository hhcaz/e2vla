import logging


level_str2flag = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARN,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


logger = logging.getLogger("SHM_TRANSPORT")
fmt = logging.Formatter("[%(levelname)s] %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)


def setup_level(level):
    if isinstance(level, str):
        level = level_str2flag[level]

    logger.setLevel(level)
    sh.setLevel(level)


setup_level(logging.INFO)


__all__ = ["setup_level"]
