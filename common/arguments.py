# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import pathlib
import os
from typing import Union

BASE_DIR = pathlib.Path(__file__).resolve(
).parent.parent   # points to 'video-to-pose3D'


class Arguments():
    # placeholder for args
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str,
                        metavar='NAME', help='target dataset')  # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb',
                        type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11',
                        type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='pretrained_h36m_detectron_coco.bin',
                        type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true',
                        help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true',
                        help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true',
                        help='save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int,
                        metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=60, type=int,
                        metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0.25,
                        type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.001,
                        type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float,
                        metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3,3',
                        type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true',
                        help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int,
                        metavar='N', help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('--subset', default=1, type=float,
                        metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int,
                        metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true',
                        help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true',
                        help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-argsimizations', action='store_true',
                        help='disable argsimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true',
                        help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true',
                        help='disable projection for semi-supervised setting')

    # Visualization
    parser.add_argument('--viz-subject', type=str,
                        metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str,
                        metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0,
                        metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str,
                        metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0,
                        metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH',
                        help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-bitrate', type=int, default=30000,
                        metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true',
                        help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1,
                        metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1,
                        metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5,
                        metavar='N', help='image size')
    # self add
    parser.add_argument('--input-npz', dest='input_npz',
                        type=str, default='', help='input 2d numpy file')
    parser.add_argument('--video', dest='input_video',
                        type=str, default='', help='input video name')

    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args


def simple_cli_args():
    parser = argparse.ArgumentParser(
        description="Videopose3D Inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=pathlib.Path,
                        help="Path to the video which should be infered.")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=BASE_DIR.parent / "outputs",
                        help="Path to which all outputs should be saved.")
    parser.add_argument("-d", "--detector_2d", type=str, default="alpha_pose",
                        help="Determine which 2d keypoint detector should be used.")

    parser.add_argument("-viz", "--visualize", action="store_false",
                        help="Disable visualization for predictions.")
    parser.add_argument("-json", "--save_json", action="store_true",
                        help="Save predications as .json")

    # Out to
    parser.add_argument("-maya", "--send_to_maya", dest="send_maya", action="store_true",
                        help="If true it trys to send the resulting joint data over a socket to maya."
                             "It assumes you have a server listining in maya for incoming command."
                             "You can find one implemention here: https://github.com/fsImageries/mayapyUtils.git")

    return parser.parse_args()


def inference_args():
    args = Arguments()

    args.dataset = "h36m"
    args.keypoints = "cpn_ft_h36m_dbb"
    args.subjects_train = "S1,S5,S6,S7,S8"
    args.subjects_test = "S9,S11"
    args.subjects_unlabeled = ""
    args.actions = "*"
    args.checkpoint = os.path.join(BASE_DIR, "checkpoint")
    args.checkpoint_frequency = 10
    args.resume = ""
    args.render = False
    args.by_subject = False
    args.export_training_curves = False
    args.stride = 1
    args.epochs = 60
    args.batch_size = 60
    args.dropout = 0.25
    args.learning_rate = 0.001
    args.lr_decay = 0.95
    args.data_augmentation = True
    args.test_time_augmentation = True
    args.architecture = "3,3,3,3,3"
    args.causal = False
    args.channels = 1024
    args.subset = 1
    args.downsample = 1
    args.warmup = 1
    args.no_eval = False
    args.dense = False
    args.disable_argsimizations = False
    args.linear_projection = False
    args.bone_length_term = True
    args.no_proj = False
    args.viz_subject = None
    args.viz_action = None
    args.viz_camera = 0
    args.viz_video = None
    args.viz_skip = 0
    args.viz_output = None
    args.viz_bitrate = 30000
    args.viz_no_ground_truth = False
    args.viz_limit = -1
    args.viz_downsample = 1
    args.viz_size = 5
    args.input_npz = ""
    args.input_video = ""
    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    return args


def alphapose_args(outputpath: pathlib.Path = None):
    args = Arguments()

    args.expID = "default"
    args.dataset = "coco"
    args.nThreads = 30
    args.debug = False
    args.snapshot = 1

    args.addDPG = False
    args.sp = False
    args.profile = False

    args.netType = "hgPRM"
    args.loadModel = None
    args.Continue = False
    args.nFeats = 256
    args.nClasses = 33
    args.nStack = 4

    args.fast_inference = True
    args.use_pyranet = True

    args.LR = 2.5e-4
    args.momentum = 0
    args.weightDecay = 0
    args.crit = "MSE"
    args.argsMethod = "rmsprop"

    args.nEpochs = 50
    args.epoch = 0
    args.trainBatch = 40
    args.validBatch = 20
    args.trainIters = 0
    args.valIters = 0
    args.init = None

    args.inputResH = 320
    args.inputResW = 256
    args.outputResH = 80
    args.outputResW = 64
    args.scale = 0.25
    args.rotate = 30
    args.hmGauss = 1

    args.baseWidth = 9
    args.cardinality = 5
    args.nResidual = 1

    args.dist = 1
    args.backend = "gloo"
    args.port = ""

    args.demo_net = "res152"
    args.inputpath = ""
    args.inputlist = ""
    args.mode = "normal"
    args.inp_dim = "608"
    args.confidence = 0.05
    args.nms_thesh = 0.6
    args.save_img = False
    args.vis = False
    args.matching = False
    args.format = ""
    args.detbatch = 1
    args.posebatch = 80

    args.video = ""
    args.webcam = "0"
    args.save_video = False
    args.vis_fast = False

    args.num_classes = 80

    args.outputpath = "examples/res/"

    if outputpath is not None:
        args.outputpath = outputpath

    return args
