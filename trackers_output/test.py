from pathlib import Path

import cv2
import torch
import os
from effdet import create_model
from effdet.config import get_efficientdet_config
from torchvision import transforms

from boxmot import DeepOcSort, ByteTrack, OcSort, HybridSort
from boxmot.utils.ops import letterbox

from trackers_output.utils import *

# Load EfficientDet model
device = torch.device('cuda:0')  # Use 'cuda' if you have a GPU

trackers = {
    'deepocsort_resnet50' : DeepOcSort(
        reid_weights=Path('resnet50_fc512_msmt17.pt'), 
        device=device,  
        cmc="none",
        cmc_off=True,
        half=False
    )
    ,
    'deepocsort_osnet' : DeepOcSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'), 
        device=device,  
        cmc="none",
        cmc_off=True,
        half=False
    )
    ,
    'bytetracker' : ByteTrack()
    ,
    'oc_sort' : OcSort()
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


def eval_detector(name):
    model_name = name
    config = get_efficientdet_config(model_name)
    model = create_model(model_name, bench_task='predict', pretrained=True).to(device)
    model.eval()
    return model, config

def run_test(model, config, tracker, reid, folder):
    c = 0
    imgs = sorted(os.listdir(BASE_DATA_DIR + folder))
    reid_folder = TRACKERS_FOLDER + tracker[0] + "/" + reid + "/"
    os.makedirs(reid_folder, exist_ok=True)
    while True:
        # Capture frame-by-frame
        # ret, frame = vid.read()

        # If ret is False, it means we have reached the end of the video
        # if not ret:
        #     break
        if c >= len(imgs):
            break 

        frame = cv2.imread(BASE_DATA_DIR + folder + '/' + imgs[c])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply letterbox resizing
        frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=config.image_size, auto=False, scaleFill=False, scaleup=False)
        
        # Preprocess frame for EfficientDet (resize and normalize)
        frame_tensor = preprocess(frame_letterbox).unsqueeze(0).to(device)

        # Perform detection
        with torch.no_grad():
            detections = model(frame_tensor)[0] 
        # Assuming detections is shaped [100, 6], with [x1, y1, x2, y2, confidence, class]
        confidence_threshold = 0.3
        
        # Filter detections based on confidence threshold
        mask = detections[:, 4] >= confidence_threshold
        filtered_dets = detections[mask]

        # Rescale coordinates from letterbox back to the original frame size
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

        # Write tracks
        write_arrays_to_file(
            reid_folder + folder + ".txt", 
            convert_tracks_to_kitti(res, c)
        )
        c += 1

        # Plot tracking results on the image
        tracker[1].plot_results(frame, show_trajectories=False)
    
        # Display the frame
        cv2.imshow('BoXMOT + EfficientDet', frame)

        # Simulate wait for key press to continue, press 'q' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    
    # Release resources
    # vid.release()
    cv2.destroyAllWindows()

    # Write seq_map
    write_arrays_to_file(
        GT_FOLDER + "evaluate_tracking.seqmap.training", 
        [[folder, 0, len(imgs) - 1, len(imgs)]],
        True
    )

if __name__ == "__main__":
    dirs = [i for i in sorted(os.listdir(BASE_DATA_DIR))]
    detectors = ['tf_efficientdet_d3'] # , 'resdet50'
    for tracker in trackers.items():
        for i, det in enumerate(detectors):
            # init detector
            model, config = eval_detector(det)
            try:
                reid_name = str(tracker[1].model.weights)
            except Exception:
                reid_name = "no-reid"
            for dir in dirs:
                run_test(model, config, tracker, reid_name, dir)
