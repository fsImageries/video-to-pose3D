from common import integration
from common.utils import Timer, add_path
from common.arguments import simple_cli_args, inference_args
from inference.inference_logic import inference, LOGGER


add_path()


def main():
    args = simple_cli_args()
    infer_args = inference_args()

    video_name = args.input.stem
    args.output = args.output / video_name

    infer_args.detector_2d = args.detector_2d
    infer_args.viz_video = str(args.input)
    infer_args.viz_output = f"{args.output}/{args.detector_2d}_{video_name}.mp4"
    infer_args.outputpath = args.output
    infer_args.outputpath.mkdir(exist_ok=True, parents=True)

    # Output Parameters
    infer_args.save_json = args.save_json
    infer_args.visualize = args.visualize
    infer_args.logger = LOGGER

    with Timer(video_name):
        preds = inference(infer_args)

    # check_outputs(args, preds)
    integration.check_outgoings(args, preds)


if __name__ == "__main__":
    main()
