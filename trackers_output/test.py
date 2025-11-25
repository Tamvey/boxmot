from pathlib import Path

import cv2
import torch
import os
import time
import numpy as np
from effdet import create_model
from effdet.config import get_efficientdet_config
from torchvision import transforms

from boxmot import DeepOcSort, ByteTrack, OcSort, HybridSort
from boxmot.utils.ops import letterbox

from trackers_output.utils import *
from collections import deque
from ultralytics import YOLO

# Load EfficientDet model
device = torch.device('cuda:0')  # Use 'cuda' if you have a GPU

trackers = {
    # 'deepocsort_resnet50' : DeepOcSort(
    #     reid_weights=Path('resnet50_fc512_msmt17.pt'), 
    #     device=device,  
    #     cmc="none",
    #     cmc_off=True,
    #     half=False
    # )
    # ,
    # 'deepocsort_osnet' : DeepOcSort(
    #     reid_weights=Path('osnet_x0_25_msmt17.pt'), 
    #     device=device,  
    #     cmc="none",
    #     cmc_off=True,
    #     half=False
    # )
    # ,
    'bytetracker' : ByteTrack()
    # ,
    # 'oc_sort' : OcSort()
}

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open the video file
# vid = cv2.VideoCapture("/media/matvey/EB6B-E36F/flash_data/animation.mp4")  # or 'path/to/your.avi'
GT_FOLDER = "/media/matvey/EB6B-E36F/data_tracking_label_2/training/"
TRACKERS_FOLDER = "/media/matvey/EB6B-E36F/data_tracking_label_2/trackers/"
TRACKER_SUB_FOLDER = "subfolder/"
BASE_DATA_DIR = "/media/matvey/EB6B-E36F/training/image_02/"


def eval_yolo(model):
    model = YOLO(f'{model}.pt')
    return model

def eval_detector(name):
    model_name = name
    config = get_efficientdet_config(model_name)
    model = create_model(model_name, bench_task='predict', pretrained=True).to(device)
    model.eval()
    return model, config

def run_test(model, image_size, tracker, reid, folder, auto_scalable=False):
    c = 0
    imgs = sorted(os.listdir(BASE_DATA_DIR + folder))
    frames = []

    for img in imgs:
        frame = cv2.imread(BASE_DATA_DIR + folder + '/' + img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames += [frame]

    reid_folder = TRACKERS_FOLDER + tracker[0] + "/" + reid + "/"
    os.makedirs(reid_folder, exist_ok=True)

    frame_times = deque(maxlen=30)  # храним последние 30 кадров
    start_time = time.time()
    while True:
        frame_start = time.time()

        if c >= len(imgs):
            break 

        frame = frames[c]

        # print(f"Read image and convert: {time.time() - frame_start}")

        # Perform detection
        with torch.no_grad():
            if auto_scalable:
                results = model(frame, verbose=False)
                if results[0].boxes is not None:
                    detections = results[0].boxes.data
                else:
                    detections = torch.empty((0, 6), device=device)
                    
            else:
                frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=image_size, auto=False, scaleFill=False, scaleup=False)
                frame_tensor = preprocess(frame_letterbox).unsqueeze(0).to(device)
                detections = model(frame_tensor)[0]

        # Assuming detections is shaped [100, 6], with [x1, y1, x2, y2, confidence, class]
        confidence_threshold = 0.3
        mask = detections[:, 4] >= confidence_threshold
        filtered_dets = detections[mask]

        # Rescale coordinates from letterbox back to the original frame size
        if not auto_scalable:
            # EfficientDet only
            filtered_dets[:, 0] = (filtered_dets[:, 0] - dw) / ratio[0]
            filtered_dets[:, 1] = (filtered_dets[:, 1] - dh) / ratio[1]
            filtered_dets[:, 2] = (filtered_dets[:, 2] - dw) / ratio[0]
            filtered_dets[:, 3] = (filtered_dets[:, 3] - dh) / ratio[1]

        # Convert class to integer and stack results
        dets = torch.cat((filtered_dets[:, :5], filtered_dets[:, 5].unsqueeze(1).int()), dim=1)

        # Convert to numpy array (N X (x, y, x, y, conf, cls))
        dets = dets.cpu().numpy()

        # Update the tracker
        res = tracker[1].update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

        frame_end = time.time()
        frame_time = frame_end - frame_start
        frame_times.append(frame_time)
        
        # current_fps = 1.0 / frame_time if frame_time > 0 else 0
        # average_fps = len(frame_times) / sum(frame_times) if frame_times else 0

        # Write tracks
        write_arrays_to_file(
            reid_folder + folder + ".txt", 
            convert_tracks_to_kitti(res, c, auto_scalable)
        )

        c += 1

        # Plot tracking results on the image
        tracker[1].plot_results(frame, show_trajectories=False)
    
        # Display the frame
        cv2.imshow('BoXMOT + EfficientDet', frame)

        # if c % 30 == 0:
        #     print(f"Frame {c}: Current FPS: {current_fps:.1f}, Average FPS: {average_fps:.1f}")

        # Simulate wait for key press to continue, press 'q' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    
    # Release resources
    # vid.release()
    total_time = time.time() - start_time
    total_fps = c / total_time if total_time > 0 else 0
    print(f"=== {folder} - {tracker[0]} ===")
    print(f"Total frames: {c}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {total_fps:.2f}")
    print("=" * 30)

    cv2.destroyAllWindows()

    # Write seq_map
    write_arrays_to_file(
        GT_FOLDER + "evaluate_tracking.seqmap.training", 
        [[folder, 0, len(imgs) - 1, len(imgs)]],
        True
    )

if __name__ == "__main__":
    dirs = [i for i in sorted(os.listdir(BASE_DATA_DIR))]

    detectors = {
        'resdet50' : eval_detector('resdet50'), 
        'tf_efficientdet_d1' : eval_detector('tf_efficientdet_d1'), 
        'tf_efficientdet_lite1' : eval_detector('tf_efficientdet_lite1'),
        'yolov8n' : eval_yolo('yolov8n')
    }

    use_detector = 'resdet50'
    for tracker in trackers.items():
        for i, det in enumerate(detectors):
            try:
                reid_name = str(tracker[1].model.weights)
            except Exception:
                reid_name = "no-reid"
            for dir in dirs:
                try:
                    run_test(detectors[use_detector][0], detectors[use_detector][1].image_size, tracker, reid_name, dir, False)
                except Exception:
                    run_test(detectors[use_detector], (640, 640), tracker, reid_name, dir, True)
