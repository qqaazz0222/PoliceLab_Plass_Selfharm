import argparse
import cv2
import copy
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
import time
from datetime import datetime
from pytz import timezone
import pymysql
from scipy.optimize import linear_sum_assignment
from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1
CCTV_ID = 28
CCTV_NAME = ''
RTSP = ''
LABEL_MAP = ["normal","biting","choking_cloth","choking_hand","hittingbody","hittinghead_floor","hittinghead_wall","kicking_wall","punching_floor","punching_wall","scratching_arm","scratching_neck","selfharm_tool"]
DEBUGMODE = False
DETECT_VAILD_TIME = 10
FPS = 10
VIDEO_DELAY = 1

def parse_args():
    parser = argparse.ArgumentParser(description='selfharm PoseC3D demo')
    # parser.add_argument('video', help='video file/url')
    # parser.add_argument('out_filename', help='output filename')
    # [Skeleton Action Recognition Config]
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_selfharm/joint.py',
        help='skeleton action recognition config file path')
    # [Skeleton Action Recognition Checkpoint]
    parser.add_argument(
        '--checkpoint',
        default='configs/posec3d/slowonly_r50_selfharm/best.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default='data/pretrain/human_detect_checkpoint.pth',
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='data/pretrain/human_pose_checkpoint.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.3,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/selfharm.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args

def detection_inference(args, model, frame):
    result = inference_detector(model, frame)
    # print("[detection_inference]", result)
    result = result[0][result[0][:, 4] >= args.det_score_thr]
    # print("[detection_inference_result]", result)
    return result

def pose_inference(args, model, frame, det_results):
    ret = []
    # for f, d in zip(frame, det_results):
    f = frame
    d = det_results
    d = [dict(bbox=x) for x in list(d)]
    pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
    # print("[pose_inference_pose]", pose)
    ret.append(pose)
    # print("[pose_inference_result]", ret)
    return ret

def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]

def get_connection(database):
    # db config
    # database = "mysql-pls"
    host = "172.30.1.46"
    port = 3306
    user = "root"
    password = "mhncity@364"
    charset = "utf8"
    # db connection
    conn = pymysql.connect(host=host,
                            port=port,
                            user=user,
                            password=password,
                            database=database,
                            charset=charset)
    return conn

def insert_log(type):
    db_insert_datas = {}
    tracking_time = datetime.now(timezone('Asia/Seoul'))
    db_insert_datas["event_date"] = copy.deepcopy(str(tracking_time)[:10]) # event_date 추가
    db_insert_datas["event_time"] = copy.deepcopy(str(tracking_time)[11:19]) # event_time 추가
    db_insert_datas["event_start"] = copy.deepcopy(str(tracking_time)[:19]) # event_start 추가
    event_start_datetime = datetime.strptime(db_insert_datas["event_start"], "%Y-%m-%d %H:%M:%S")
    event_end_datetime = event_start_datetime
    db_insert_datas["event_end"] = copy.deepcopy(event_end_datetime)
    db_insert_datas["event_clip_directory"] = copy.deepcopy( ("/") )
    # DB 연결
    db_connection = get_connection("mysql-pls")
    print(f"┌────────────────────[SELFHARM DETECT]────────────────────┐")
    print(f"│                  Log Store In Databse                   │")
    print(f"└─────────────────────────────────────────────────────────┘")
    try:
        with db_connection.cursor() as cursor:
            sql = """
            INSERT INTO event 
            (cctv_id, event_type, event_location, event_detection_people, 
            event_date, event_time, event_clip_directory, 
            event_confirm_date, event_confirm_time, event_check, 
            event_start, event_end) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # 현재 날짜 및 시간 가져오기, 이건 이벤트 확인 시간, 날짜에 사용되므로, 차후 수정 필요
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            # DB에 삽입할 값 설정
            values = [CCTV_ID,"selfharm", CCTV_NAME, type,  db_insert_datas['event_date'], db_insert_datas["event_time"],
                        db_insert_datas["event_clip_directory"], current_date, current_time, 0,
                        db_insert_datas["event_start"], db_insert_datas["event_end"]]

            cursor.execute(sql, values)
            
        db_connection.commit()
        last_db_insert_time = current_time 
        print(" * Store Successfully")
    except Exception as e:
        print(f" * Error Occurred: {e}")
    finally:
        db_connection.close() 

def main():
    args = parse_args()
    print()
    print(f"┌────────────────────[SELFHARM DETECT]────────────────────┐")
    print(f"│                 Loading Models & Weight                 │")
    print(f"└─────────────────────────────────────────────────────────┘")
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    model = init_recognizer(config, args.checkpoint, args.device)
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    det_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    print()
    print(f"┌────────────────────[SELFHARM DETECT]────────────────────┐")
    print(f"│                Loading CCTV Stream Video                │")
    print(f"└─────────────────────────────────────────────────────────┘")
    print(f"* cctv ID: {CCTV_ID}")
    print(f"* cctv name: {CCTV_NAME}")
    print(f"* rtsp url: {RTSP}")
    vid = cv2.VideoCapture(RTSP)
    selfharm_count = 0
    selfharm_last_detect = datetime.now()
    prev_time = 0

    print()
    print(f"┌────────────────────[SELFHARM DETECT]────────────────────┐")
    print(f"│                       Detect Start                      │")
    print(f"└─────────────────────────────────────────────────────────┘")
    while True:
        now = datetime.now()
        flag, frame = vid.read()
        current_time = time.time() - prev_time
        if (flag is True) and (current_time > 1./ FPS) :
            prev_time = time.time()
            h, w, _ = frame.shape
            det_results = None
            num_human = None
            det_results = detection_inference(args, det_model, frame)
            num_human = len(det_results)
            if num_human == 0:
                print(f"[{now}][HUMAN DETECT] No Human Detect")
                time.sleep(VIDEO_DELAY)
            else:
                print(f"[{now}][HUMAN DETECT] {num_human} Human Detect")
                pose_results = pose_inference(args, pose_model, frame, det_results)
                
                num_frame = 1
                fake_anno = dict(
                    frame_dir='',
                    label=-1,
                    img_shape=(h, w),
                    original_shape=(h, w),
                    start_index=0,
                    modality='Pose',
                    total_frames=num_frame)

                if GCN_flag:
                    # We will keep at most `GCN_nperson` persons per frame.
                    tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
                    keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
                    fake_anno['keypoint'] = keypoint
                    fake_anno['keypoint_score'] = keypoint_score
                else:
                    num_person = max([len(x) for x in pose_results])
                    # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
                    num_keypoint = 17
                    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                        dtype=np.float16)
                    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                            dtype=np.float16)
                    for i, poses in enumerate(pose_results):
                        for j, pose in enumerate(poses):
                            pose = pose['keypoints']
                            keypoint[j, i] = pose[:, :2]
                            keypoint_score[j, i] = pose[:, 2]
                    fake_anno['keypoint'] = keypoint
                    fake_anno['keypoint_score'] = keypoint_score

                action_label = ''
                action_acc = 0.0
                if fake_anno['keypoint'] is not None:
                    temp = []
                    results = inference_recognizer(model, fake_anno)
                    if DEBUGMODE:
                        for result in results:
                            temp.append({LABEL_MAP[result[0]]: round(float(result[1]), 4)})
                        print(f" ├─[RESULTS] {temp}")
                    action_label = LABEL_MAP[results[0][0]]
                    action_acc = results[0][1]
                print(f" └─[POSE DETECT] ACTION: {action_label} / P: {action_acc}")
                # selfharm 오탐 방지
                if action_label != "normal":
                    diff = now - selfharm_last_detect
                    if diff.seconds > DETECT_VAILD_TIME:
                        selfharm_count = 0
                        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
                    else:
                        if selfharm_count == 0:
                            insert_log(num_human)
                        selfharm_count += 1
                    if selfharm_count > 10:
                        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
                    selfharm_last_detect = now

if __name__ == '__main__':
    # DB 테스트
    # insert_log(0)
    DEFAULT_CCTV_ID = 28
    RTSP_DICT = {
        28: {'name': '엣지카메라 현관', 'url': 'rtsp://admin:admin@172.30.1.41/stream1'},
        29: {'name': '엣지카메라 에어컨 옆', 'url': 'rtsp://admin:wonwoo0!23@172.30.1.42/stream1'},
        30: {'name': '엣지카메라 테이블', 'url': 'rtsp://admin:admin@172.30.1.43/stream1'},
        101: {'name': '충정로 연구소', 'url': 'rtsp://admin:dpadpdlcl-1@172.30.1.2:554/cam/realmonitor?channel=1&subtype=0'}
        }
    if CCTV_ID:
        RTSP = RTSP_DICT[CCTV_ID]['url']
        CCTV_NAME = RTSP_DICT[CCTV_ID]['name']
    else:
        RTSP = RTSP_DICT[DEFAULT_CCTV_ID]['url']
        CCTV_NAME = RTSP_DICT[DEFAULT_CCTV_ID]['name']
    main()
