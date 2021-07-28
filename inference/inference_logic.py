import os
import cv2
import time
import pathlib

# from ..common import visualization
from common import visualization
from common.arguments import inference_args, alphapose_args, BASE_DIR
from common.camera import *
from common.generators import UnchunkedGenerator
from common.visualization import render_animation
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path, get_device
from common.logger import file_logger
from typing import Union
from pyUtils import nphelper


# ---------------------------- Project Setup -------------------------- #
# --------------------------------------------------------------------- #

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {"layout_name": "coco", "num_joints": 17,
            "keypoints_symmetry": [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()

LOGGER = file_logger(log_dir=BASE_DIR.parent / "logs",
                     log_name=pathlib.Path(__file__).stem)

# ------------------------------ Helpers ------------------------------ #
# --------------------------------------------------------------------- #


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def ckpt_time(ckpt=None):
    # record time
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


def get_detector_2d(detector_name, *args):
    def get_alpha_pose():
        import opt  # TODO Fix import madness in all .py files
        alpha_opt = alphapose_args(*args)
        file_name = pathlib.Path(__file__).stem
        # alpha_opt.logger = file_logger(
        #     log_dir=BASE_DIR / "logs", log_name=file_name)
        alpha_opt.logger = LOGGER
        opt.opt = alpha_opt
        import joints_detectors.Alphapose.gene_npz as alpha_pose
        # alpha_pose.set_logger()
        return alpha_pose.generate_kpts

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose  # TODO Fix cuda() device assignments

    detector_map = {
        "alpha_pose": get_alpha_pose,
        "hr_pose": get_hr_pose,
        # "open_pose": open_pose
    }
    assert detector_name in detector_map, f"2D detector: {detector_name} not implemented yet!"
    return detector_map[detector_name]()


def get_default_args(videopath: Union[str, pathlib.Path]):
    args = inference_args()

    if isinstance(videopath, str):
        videopath = pathlib.Path(videopath)
    filepath = pathlib.Path(__file__)

    args.detector_2d = "alpha_pose"
    args.output = BASE_DIR.parent / "outputs" / videopath.stem
    args.output.mkdir(exist_ok=True, parents=True)
    args.viz_output = f"{args.output}/{args.detector_2d}_{videopath.stem}.mp4"
    args.viz_video = str(videopath)
    args.outputpath = args.output

    # Output Parameters
    args.save_json = True
    args.visualize = True
    # args.logger = file_logger(
    #     log_dir=BASE_DIR / "logs", log_name=filepath.stem)
    args.logger = LOGGER

    return args


# ----------------------------- Inference ----------------------------- #
# --------------------------------------------------------------------- #

def get_keypoints(detector_2d, video, input_npz=None):

    # 2D kpts loads or generate
    if not input_npz:
        video_name = video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(input_npz)
        keypoints = npz["kpts"]  # (N, 17, 2)

    keypoints_symmetry = metadata["keypoints_symmetry"]
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(
        [4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  Suppose using the camera parameter
    keypoints = normalize_screen_coordinates(
        keypoints[..., :2], w=1000, h=1002)

    return {"kpts": keypoints, "kpts_lr": [kps_left, kps_right], "jnts_lr": [joints_left, joints_right]}


def get_model(checkpoint, channels, resume, causal, dropout, dense, evalnet, logger=None):
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3],
                              causal=causal, dropout=dropout,
                              channels=channels, dense=dense)

    # if torch.cuda.is_available():
    #     model_pos = model_pos.cuda()
    model_pos.to(get_device())

    # load trained model
    chk_filename = os.path.join(checkpoint, resume if resume else evalnet)

    if logger:
        logger.info(f"Loading checkpoint {chk_filename}")

    checkpoint = torch.load(
        chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint["model_pos"])

    return model_pos


def set_output(preds, output, save_json=None):
    # save 3D joint points
    points_out = os.path.join(output, "3d_joint_positions")
    np.save(f"{points_out}.npy", preds, allow_pickle=True)

    if save_json:
        nphelper.np2json(preds, f"{points_out}.json")


def visualize(preds, in_kpts, output, video, fps, bitrate, limit, downsample, size, skip):
    rot = np.array([0.14070565, -0.15007018, -0.7552408,
                   0.62232804], dtype=np.float32)
    preds = camera_to_world(preds, R=rot, t=0)

    # We don"t have the trajectory, but at least we can rebase the height
    preds[:, :, 2] -= np.min(preds[:, :, 2])
    anim_output = {"Reconstruction": preds}
    in_kpts = image_coordinates(
        in_kpts[..., :2], w=1000, h=1002)

    if not output:
        output = "outputs/alpha_result.mp4"

    render_animation(in_kpts, anim_output, Skeleton(),
                     fps, bitrate, np.array(70., dtype=np.float32),
                     output, limit=limit, downsample=downsample,
                     size=size, input_video_path=video,
                     viewport=(1000, 1002), input_video_skip=skip)


def inference(args):
    time0 = ckpt_time()

    detector_2d = get_detector_2d(args.detector_2d, args.outputpath)
    assert detector_2d, "detector_2d should be in ({alpha, hr, open}_pose)"
    kpts = get_keypoints(detector_2d, args.viz_video, args.input_npz)

    ckpt, time1 = ckpt_time(time0)
    args.logger.info(
        "-------------- load data spends {:.2f} seconds".format(ckpt))

    model_pos = get_model(args.checkpoint, args.channels, args.resume,
                          args.causal, args.dropout, args.dense, args.evaluate, args.logger)

    ckpt, time2 = ckpt_time(time1)
    args.logger.info(
        "-------------- load 3D model spends {:.2f} seconds".format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    # Estimating
    input_keypoints = kpts["kpts"].copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kpts["kpts_lr"][0], kps_right=kpts["kpts_lr"][1],
                             joints_left=kpts["jnts_lr"][0], joints_right=kpts["jnts_lr"][1])
    prediction = evaluate(gen, model_pos, return_predictions=True)

    # Saving Output
    set_output(prediction, args.outputpath, args.save_json)

    ckpt, time3 = ckpt_time(time2)
    args.logger.info(
        "-------------- generate reconstruction 3D data spends {:.2f} seconds".format(ckpt))

    if args.visualize:
        visualization.logger = args.logger
        fps = cv2.VideoCapture(args.viz_video).get(cv2.CAP_PROP_FPS)
        visualize(prediction, input_keypoints,
                  args.viz_output, args.viz_video,
                  fps, args.viz_bitrate, args.viz_limit,
                  args.viz_downsample, args.viz_size, args.viz_skip)

    ckpt, _ = ckpt_time(time3)
    args.logger.info("total spend {:2f} second".format(ckpt))

    return prediction


if __name__ == "__main__":
    import sys

    in_file = sys.argv[1]

    args = get_default_args(in_file)
    inference(args)

    # file_logger(pathlib.Path("/Users/joshua/Desktop/video-to-pose3D/logs"))
