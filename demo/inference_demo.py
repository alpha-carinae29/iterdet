from mmdet.apis import init_detector, inference_detector
import mmcv
import argparse
import cv2 as cv
import os
import time


def build_model(args):
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print("============================================================")
    print("             The model has been loaded")
    print("============================================================")
    return model

def inference_and_save_results(model, args):
    input_cap = cv.VideoCapture(args.video_file)
    if (input_cap.isOpened()):
        print('opened video ', args.video_file)
    frame_num = 0
    w = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    if args.result_video_path:
        if not os.path.exists(args.result_video_path):
            os.makedirs(args.result_video_path)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(os.path.join(args.result_video_path,"iterdet_results.avi"),fourcc, 20.0, (960,540))
        print("The result video will be saved on", args.result_video_path)
    print("============================================================")
    print("             Start Inference and Save Results")
    print("============================================================")
    while (frame_num <= args.frame and input_cap.isOpened()):
        ret, frame = input_cap.read()
        t_begin = time.perf_counter()
        result = inference_detector(model, frame)
        t_end = time.perf_counter()
        with open(os.path.join(args.results_path, str(frame_num) + ".txt"), "w") as file:
            for box in result[0]:
                score = box[-1]
                if score > args.confidence_thresh:
                    obj_data = ["pedestrian"]
                    obj_data.append(str(score))
                    obj_data.extend([str(int(i)) for i in box[:-1]])
                    file.write(" ".join(obj_data) + "\n")
                    if args.result_video_path:
                        x0 = int(obj_data[2])
                        y0 = int(obj_data[3])
                        x1 = int(obj_data[4])
                        y1 = int(obj_data[5])
                        x_min = max(0, x0)
                        y_min = max(0, y0)
                        x_max = min(w, x1)
                        y_max = min(h, y1)
                        color = (255,0,0)
                        cv.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                        cv.rectangle(frame, (x0, y0), (x0 + 100, y0 - 30), color, -1)
                        cv.putText(frame,
                        str(obj_data[0]),
                        (x0, y0),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)
        if args.result_video_path:
            out.write(cv.resize(frame,(960,540)))
            
        print("frame number {} processed in {} seconds".format(frame_num, round(t_end-t_begin, 2)), end="\r")
        frame_num += 1
    print("+++++++++++++++++++++++++Finished+++++++++++++++++++++++++++++++++++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", required=True, type=str, help="path to the video file")
    parser.add_argument("--config_file", required=True, type=str, help="path to the config file")
    parser.add_argument("--checkpoint_file", required=True, type=str, help="path to the checkpoint file")
    parser.add_argument("--results_path", required=True, type=str, help="where to save inference results")
    parser.add_argument("--confidence_thresh", default=0.5, type=float, help="minimum confidence score to consider a box as prediction")
    parser.add_argument("--frame", default=4500, type=int, help="how many frames use for inference")
    parser.add_argument('--result_video_path', default = "", type=str, help="path to result video file")
    args = parser.parse_args()
    model = build_model(args)
    inference_and_save_results(model, args)
