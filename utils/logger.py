import logging
from enum import Enum
from logging import config, getLogger

import coloredlogs
import yaml


class LogLevel(int, Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def init_logger(
    config_file: str = None,
    name: str = "",
    level: LogLevel | str = LogLevel.INFO,
) -> logging.Logger:
    if type(level) is str:
        level = LogLevel[level.upper()]

    coloredlogs.install(level)
    if config_file is not None:
        config.dictConfig(yaml.load(open(config_file).read(), Loader=yaml.SafeLoader))

    logger = logging.getLogger(name)

    return logger
