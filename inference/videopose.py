import os
import cv2
import time
import common.arguments as arg_parsers

from common.camera import *
from common.generators import UnchunkedGenerator
from common.visualization import render_animation
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path, np2json


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {"layout_name": "coco", "num_joints": 17, "keypoints_symmetry": [
    [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name, *args):
    def get_alpha_pose():
        import opt
        alpha_opt = arg_parsers.alphapose_args(*args)
        opt.opt = alpha_opt
        import joints_detectors.Alphapose.gene_npz as alpha_pose
        return alpha_pose.generate_kpts

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        "alpha_pose": get_alpha_pose,
        "hr_pose": get_hr_pose,
        # "open_pose": open_pose
    }

    assert detector_name in detector_map, f"2D detector: {detector_name} not implemented yet!"

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    detector_2d = get_detector_2d(args.detector_2d, args.outputpath)

    assert detector_2d, "detector_2d should be in ({alpha, hr, open}_pose)"

    # 2D kpts loads or generate
    if not args.input_npz:
        video_name = args.viz_video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz["kpts"]  # (N, 17, 2)

    keypoints_symmetry = metadata["keypoints_symmetry"]
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(
        [4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  Suppose using the camera parameter
    keypoints = normalize_screen_coordinates(
        keypoints[..., :2], w=1000, h=1002)

    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                              dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print("-------------- load data spends {:.2f} seconds".format(ckpt))

    # load trained model
    chk_filename = os.path.join(
        args.checkpoint, args.resume if args.resume else args.evaluate)
    print("Loading checkpoint", chk_filename)
    checkpoint = torch.load(
        chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint["model_pos"])

    ckpt, time2 = ckpt_time(time1)
    print("-------------- load 3D model spends {:.2f} seconds".format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print("Rendering...")
    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)

    # save 3D joint points
    points_out = os.path.join(args.outputpath, "3d_joint_positions")
    np.save(f"{points_out}.npy", prediction, allow_pickle=True)

    if args.save_maya:
        np2json(prediction, f"{points_out}.json")

    rot = np.array([0.14070565, -0.15007018, -0.7552408,
                   0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don"t have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    anim_output = {"Reconstruction": prediction}
    input_keypoints = image_coordinates(
        input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print(
        "-------------- generate reconstruction 3D data spends {:.2f} seconds".format(ckpt))

    if not args.viz_output:
        args.viz_output = "outputs/alpha_result.mp4"

    fps = cv2.VideoCapture(args.viz_video).get(cv2.CAP_PROP_FPS)
    render_animation(input_keypoints, anim_output,
                     Skeleton(), fps, args.viz_bitrate, np.array(
                         70., dtype=np.float32), args.viz_output,
                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(1000, 1002),
                     input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print("total spend {:2f} second".format(ckpt))


def cli_main():
    """
    Do image -> 2d points -> 3d points to video.
    :return: None
    """
    args = arg_parsers.simple_cli_args()
    infer_args = arg_parsers.inference_args()

    infer_args.detector_2d = args.detector_2d
    video_name = args.input.stem
    infer_args.viz_video = str(args.input)
    infer_args.viz_output = f"{args.output}/{args.detector_2d}_{video_name}.mp4"
    infer_args.outputpath = args.output

    # Output Parameters
    infer_args.save_maya = args.save_maya
    infer_args.send_maya = args.send_maya
    infer_args.viz = args.viz


    with Timer(args.input):
        main(infer_args)


if __name__ == "__main__":
    cli_main()
