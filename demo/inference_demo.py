from mmdet.apis import init_detector, inference_detector
import mmcv
import argparse

def build_model(args):
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", required=True, type=str, help="path to the video file")
    parser.add_argument("--config_file", required=True, type=str, help="path to the config file")
    parser.add_argument("--checkpoint_file", required=True, type=str, help="path to the checkpoint file")
    args = parser.parse_args()
    model = build_model(args)
    print(model)
