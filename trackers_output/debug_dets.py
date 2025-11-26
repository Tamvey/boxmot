import cv2, os



GT_FOLDER = "/media/matvey/EB6B-E36F/data_tracking_label_2/training/label_02/"
TRACKERS_FOLDER = "/media/matvey/EB6B-E36F/data_tracking_label_2/trackers/"
BASE_DATA_DIR = "/media/matvey/EB6B-E36F/training/image_02/"

def try_float(x):
    try: 
        return float(x) 
    except Exception: 
        return x

def read_rects(paths_to_files):
    res = []
    with open(paths_to_files, "r") as fr:
        for line in fr:  
            res.append(list(map(lambda x: try_float(x), line.strip().split(" "))))
    return res

def run_test(dirs):
    for dir in dirs:
        # Get initial imgs
        imgs = [BASE_DATA_DIR + dir + "/" + i for i in sorted(os.listdir(BASE_DATA_DIR + dir))]

        # Read trackers
        for tracker_path in [TRACKERS_FOLDER + tracker + "/" for tracker in os.listdir(TRACKERS_FOLDER)]:
            # Reids paths in tracker
            reid_paths = [
                tracker_path + reid + "/"
                for reid in os.listdir(tracker_path) 
                if os.path.isdir(os.path.join(tracker_path, reid))
            ]
            for reid in reid_paths:
                os.makedirs(reid + "debug/", exist_ok=True)
                got_rects = read_rects(reid + dir + ".txt")
                gt_rects = read_rects(GT_FOLDER + "/" + dir + ".txt")
                for i, img_path in enumerate(imgs):
                    print(img_path)
                    img = cv2.imread(img_path)
                    # Get rects per image
                    got_img_rects = list(filter(lambda line: line[0] == i, got_rects))
                    gt_img_rects = list(filter(lambda line: line[0] == i, gt_rects))
                    # Draw rects
                    for rect in got_img_rects:
                        img = cv2.rectangle(img, (int(rect[6]), int(rect[7])), (int(rect[8]), int(rect[9])), (0, 0, 255), 1) # red
                    for rect in gt_img_rects:
                        img = cv2.rectangle(img, (int(rect[6]), int(rect[7])), (int(rect[8]), int(rect[9])), (0, 255, 0), 1) # green

                    # Folder kind of 0000, 0001
                    folder = reid + "debug/" + os.path.basename(os.path.dirname(img_path)) + '/'
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(folder + os.path.basename(img_path), img)               


if __name__ == "__main__":
    dirs = [i for i in sorted(os.listdir(BASE_DATA_DIR))]
    run_test(dirs)
