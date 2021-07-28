from inference.inference_logic import get_default_args, inference
import sys
import pathlib
from common import logger
from common.utils import add_path

add_path()

# from joints_detectors.Alphapose import gene_npz


# logger.file_logger(pathlib.Path("/Users/joshua/Desktop/video-to-pose3D/logs"))
# BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
# print(BASE_DIR)

in_file = sys.argv[1]

args = get_default_args(in_file)
inference(args)

# BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# print(pathlib.Path(__file__).stem)
