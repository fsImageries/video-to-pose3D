import argparse
import pathlib
import pickle

from common.logger import file_logger
from common.arguments import alphapose_args


opt = None
THIS_DIR = pathlib.Path(__file__).parent


def get_opt():

    with open(THIS_DIR/"opt.pkl", "rb") as inf:
        args = pickle.load(inf)
    log_name = args.logger.name
    args.logger = file_logger(log_name=log_name)
    return args
