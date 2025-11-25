RESDET50_TO_KITTI_SIMPLE = {
    1: 'Pedestrian',   # person
    2: 'Cyclist',      # bicycle
    3: 'Car',          # car
    4: 'Car',          # motocycle
    6: 'Van',          # bus
    7: 'Tram',
    8: 'Truck',        # truck
}

def convert_tracks_to_kitti(dets, frame):
    """
        got: (x, y, x, y, id, conf, cls, ind)
        return: kitti format dets (check formats.txt)
    """
    res = []
    for det in dets:
        x1, y1, x2, y2, track_id, conf, cls_idx, ind = det
        clss = "Misc" if cls_idx not in RESDET50_TO_KITTI_SIMPLE.keys() else RESDET50_TO_KITTI_SIMPLE[cls_idx]
        print(f"Detector res: {cls_idx}, In KITTI: {clss}\n\n")
        res += [
            [
                frame,
                track_id, 
                clss,
                0,  
                0,
                -10,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                conf
            ]
        ]
    return res

def write_arrays_to_file(filepath, arrays, csv=False):
    delim = "," if csv else " "
    for array in arrays:
        with open(filepath, "a") as fw:
            fw.write(
                delim.join(map(lambda x: str(x), array)) + "\n"
            )