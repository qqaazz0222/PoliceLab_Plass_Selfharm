import os
import random
import cv2
import json

path = "./selfharm"
vid_path = "./selfharm/rgb"

divide = [9, 1]
train = []
test = []
key = [
    "vid_name"
    "label",
    "start_frame",
    "end_frame",
]
labels = ['normal', 'biting', 'choking_cloth', 'choking_hand', 'hittingbody', 'hittinghead_floor', 'hittinghead_wall', 'kicking_wall', 'punching_floor', 'punching_wall', 'scratching_arm', 'scratching_neck', 'selfharm_tool']
file_list = os.listdir(vid_path)
idx = 0
last_idx = len(file_list) - 1
for file in file_list:
    cap = cv2.VideoCapture(os.path.join(vid_path, file))
    temp = {}
    label = 0
    name = file.split(".")[0]
    for l in labels:
        if l in name:
            label = labels.index(l)
            break
    temp["vid_name"] = name
    temp["label"] = label
    temp["start_frame"] = 0
    temp["end_frame"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{round(idx/last_idx*100, 2)}%]{temp}")
    rdn = random.randrange(0,divide[0]+divide[1])
    if rdn < divide[0]:
        train.append(temp)
    else:
        test.append(temp)
    idx += 1
    
with open(os.path.join(path, "train.json"), 'w', encoding='utf-8') as make_file:
    json.dump(train, make_file, indent="\t")
with open(os.path.join(path, "test.json"), 'w', encoding='utf-8') as make_file:
    json.dump(test, make_file, indent="\t")
    