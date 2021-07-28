import inspect
import ntpath
import shutil
import os

import numpy as np
import torch

from tqdm import tqdm
from common.utils import calculate_area, get_device
from common.logger import file_logger
from common.arguments import alphapose_args
from joints_detectors.Alphapose.opt import get_opt
from .SPPE.src import main_fast_inference
from .dataloader import DetectionLoader, DetectionProcessor, DataWriter, Mscoco, VideoLoader
from .SPPE.src.main_fast_inference import *
from .pPose_nms import write_json
from .fn import getTime


opt = get_opt()

args = opt
args.dataset = 'coco'
args.fast_inference = False
args.save_img = True
args.sp = True
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

main_fast_inference.logger = args.logger

logger = None


def set_args(*in_args):
    global args

    args = alphapose_args(*in_args)
    args.dataset = 'coco'
    args.fast_inference = False
    args.save_img = True
    args.sp = True
    if not args.sp:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')


def set_logger(in_logger=None):
    global logger

    if in_logger is None:
        _frame = inspect.stack()[1]
        _module = inspect.getmodule(_frame[0])
        _filename = _module.__file__
        log_name = os.path.basename(_filename).rsplit(".", 1)[0]
        logger = file_logger(log_name=log_name)
    else:
        logger = in_logger


def model_load():
    model = None
    return model


def image_interface(model, image):
    pass


def generate_kpts(video_file):
    final_result, video_name = handle_video(video_file)

    # ============ Changing ++++++++++

    kpts = []
    no_person = []
    for i in range(len(final_result)):
        if not final_result[i]['result']:  # No people
            no_person.append(i)
            kpts.append(None)
            continue

        kpt = max(final_result[i]['result'],
                  key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

        kpts.append(kpt.data.numpy())

        for n in no_person:
            kpts[n] = kpts[-1]
        no_person.clear()

    for n in no_person:
        kpts[n] = kpts[-1] if kpts[-1] else kpts[n-1]

    # ============ Changing End ++++++++++

    name = f'{args.outputpath}/{video_name}.npz'
    kpts = np.array(kpts).astype(np.float32)
    args.logger.info(f"kpts npz save in {name}")
    # print('kpts npz save in ', name)
    np.savez_compressed(name, kpts=kpts)

    return kpts


def handle_video(video_file):
    # =========== common ===============
    args.video = video_file
    base_name = os.path.basename(args.video)
    video_name = base_name[:base_name.rfind('.')]

    args.outputpath = os.path.join(args.outputpath, f'alpha_pose_{video_name}')
    if os.path.exists(args.outputpath):
        shutil.rmtree(f'{args.outputpath}/vis', ignore_errors=True)
    else:
        os.mkdir(args.outputpath)
    # =========== end common ===============

    # =========== video ===============
    videofile = args.video
    mode = args.mode
    if not len(videofile):
        raise IOError('Error: must contain --video')
    # Load input video
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()
    args.logger.info(f"the video is {fps} f/s")
    # print('the video is {} f/s'.format(fps))
    # =========== end video ===============
    # Load detection loader
    args.logger.info("Loading YOLO model...")
    # print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    #  start a thread to read frames from the file video stream
    det_processor = DetectionProcessor(det_loader).start()
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.to(get_device())
    # pose_model.cuda()
    pose_model.eval()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_' +
                             ntpath.basename(video_file).split('.')[0] + '.avi')
    # writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()
    writer = DataWriter(args.save_video, outpath=args.outputpath).start()
    args.logger.info("Start pose estimation...")
    # print('Start pose estimation...')
    im_names_desc = tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:

        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores,
             pt1, pt2) = det_processor.read()
            if orig_img is None:
                args.logger.info(f"{i}-th image read None: handle_video")
                # print(f'{i}-th image read None: handle_video')
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None,
                            orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if datalen % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                # inps_j = inps[j *
                #               batchSize:min((j + 1) * batchSize, datalen)].cuda()
                inps_j = inps[j *
                              batchSize:min((j + 1) * batchSize, datalen)].to(get_device())
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2,
                        orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )
    if (args.save_img or args.save_video) and not args.vis_fast:
        args.logger.info(
            "===========================> Rendering remaining images in the queue...")
        args.logger.info(
            "===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).")

        # print('===========================> Rendering remaining images in the queue...')
        # print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while writer.running():
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)

    return final_result, video_name


if __name__ == "__main__":
    os.chdir('../..')
    print(os.getcwd())

    # handle_video(img_path='outputs/image/kobe')
    generate_kpts('outputs/dance.mp4')
