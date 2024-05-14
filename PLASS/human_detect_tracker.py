import os
import cv2
import torch
import argparse
import openpyxl
import numpy as np
from numpy import random
import datetime
import sys
import multiprocessing
from datetime import timedelta
from dbConnector import getConnection
import copy
from BoTSORT.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from BoTSORT.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from BoTSORT.yolov7.models.experimental import attempt_load
from BoTSORT.yolov7.utils.datasets import LoadStreams, LoadImages
from BoTSORT.tracker.mc_bot_sort import BoTSORT
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn
from pytz import timezone

sys.path.insert(0, '../BoTSORT/yolov7')

TIME_ZONE = timezone('Asia/Seoul')
conn = getConnection("mysql-pls")
cctv = {}  # cctv 설정값 저장할 변수
    
def create_folder(path):
    """
    Creates a new folder at the specified path if it doesn't already exist.
    """
    try:
        # Check if the folder already exists
        if not os.path.exists(path):
            os.makedirs(path)
            return f"Folder created at: {path}"
        else:
            return f"Folder already exists at: {path}"
    except Exception as e:
        return f"An error occurred: {e}"

def blockPrint():
    global backupstdout
    backupstdout=sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    global backupstdout
    sys.stdout =backupstdout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./model/human_detect_tracker_checkpoint.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)') # 480 -> 1080 수정완료.
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --class 0, or --6class 0 2 3') # 0몸통, 1얼굴, 같이 쓰려면 [0,1]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save_snapshot', action='store_true', default=True, help='save snapshots for each detected object')
    parser.add_argument('--snapshot_dir', type=str, default=f'/home/mhncity/data/person/27/', help='directory for saving snapshots')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.7, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.07, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.9, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=100, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.49, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    args = parser.parse_args()
    return args

def worker(queue, cctv_info):
    blockPrint()
    args = parse_args()
    args.jde = False
    args.ablation = False
    cctv_id, cctv_ip, cctv_name = cctv_info['cctv_id'], cctv_info['ip'], cctv_info['name']
    device = 'cuda'
    source, weights, view_img, save_txt, imgsz, trace = cctv_info['ip'], args.weights, args.view_img, args.save_txt, args.img_size, args.trace
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Initialize
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    if trace:
        model = TracedModel(model, device, args.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker 
    tracker = BoTSORT(frame_rate=30.0)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    ################# Run inference ###################
    loop_count=0
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32   
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=args.augment)[0]

        # Apply NMS X
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

        # Initialize det variable
        det = []

        # Apply Classifier X
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        face_boxes = []
        body_boxes = []
        human_boxes = []
        id_array=[]
        
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            

            # Run tracker
            detections = []
            if len(det):
                # print(f"감지된 객체 수 (몸통, 머리를 개별 객체로 봄을 주의.) : {len(det)}")
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes
            
            online_targets = tracker.update(detections)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                det_score = t.score
                
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    # print(tlwh)
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    if tcls == 0: # 몸통
                        body_boxes.append([x1,y1,x2,y2,tid,tcls])
                        human_boxes.append(np.array([x1,y1,x2,y2, 1]))
                        id_array.append(tid)
                    elif tcls == 1: # 얼굴
                        face_boxes.append([x1,y1,x2,y2,tid,tcls])
                    
                    online_tlwhs.append([x1, y1, x2, y2])
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)
            
            # 메인 쓰레드로 값 전달
            queue.put({"ids": id_array, "boxes": np.array(human_boxes)})
            


if __name__ == '__main__':
    cctv_info = {'cctv_id': 101, 'ip': 'rtsp://admin:dpadpdlcl-1@172.30.1.2:554/cam/realmonitor?channel=1&subtype=0', 'name': '충정로 연구소'}
    worker(cctv_info)

    