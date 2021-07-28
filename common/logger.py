import inspect
import logging
import pathlib
import datetime
import sys
import os

from typing import Union
from common.arguments import BASE_DIR


def file_logger(log_dir: Union[str, pathlib.Path] = None, log_name: str = None) -> logging.Logger:
    if log_dir is None:
        log_dir = BASE_DIR / "logs"

    if isinstance(log_dir, str):
        log_dir = pathlib.Path(log_dir)

    if log_name is None:
        _frame = inspect.stack()[1]
        _module = inspect.getmodule(_frame[0])
        _filename = _module.__file__
        log_name = os.path.basename(_filename).rsplit(".", 1)[0]

    log_dir = log_dir / log_name
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    date_stamp = datetime.date.today().strftime("%d.%m.%Y")
    handler = logging.FileHandler(log_dir / f"{log_name}_{date_stamp}.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

