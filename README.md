# Dongguk Police Lab Selfharm Module

## Installation

### 1. Creating Docker Environment from Docker File
```
git clone https://github.com/qqaazz0222/PoliceLab_Plass_Selfharm.git
cd PoliceLab_Plass_Selfharm
docker build --tag plass-module:1.0 .
docker run --gpus all -it --name plass-selfharm plass-module:1.0
```

### 2. Add Database Information
Target: ./dbConnector.py

> Change Here!

```
host = "" #input host address
port = 3306 #change port(optinal)
user = "" #input database user
password = "" #input database password
charset = "utf8" #change charset(optinal)
```
### 3. Run Module
```
cd PLASS
python selfharm.py
```
## Donwload Checkpoint Files

### Selfharm Detect Checkpoint

File : PLASS/model/selfharm_pose_checkpoint_best.pth<br/>
Link : [Download 🔗](https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EepW2QK6ygJMpTSpk7zsiXoBo3NSq4KibNp-tAgOsCNRDA?e=EqOkKb)

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EepW2QK6ygJMpTSpk7zsiXoBo3NSq4KibNp-tAgOsCNRDA?e=EqOkKb&download=1" -O PLASS/model/selfharm_pose_checkpoint_best.pth
```

### Human Pose Estimation Checkpoint

File : PLASS/model/human_pose_checkpoint.pth<br/>
Link : [Download 🔗](https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EaKk7gy9dq5AgyaBNK7gVcYByDjnz2mK7eQ0wyrPYdstug?e=1latwR)

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EaKk7gy9dq5AgyaBNK7gVcYByDjnz2mK7eQ0wyrPYdstug?e=1latwR&download=1" -O PLASS/model/human_pose_checkpoint.pth
```

### Human Detect Estimation Checkpoint

File : PLASS/model/human_detect_checkpoint.pth<br/>
Link : [Download 🔗](https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ebk9j6XmHZNKkjTB95HXLOgBK856p99nlb2jMOeuKkeUYg?e=H3znvQ)

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ebk9j6XmHZNKkjTB95HXLOgBK856p99nlb2jMOeuKkeUYg?e=H3znvQ&download=1" -O PLASS/model/human_detect_checkpoint.pth
```

### Human Detect Tracker Estimation Checkpoint

File : PLASS/model/human_detect_tracker_checkpoint.pt<br/>
Link : [Download 🔗](https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZypUOJgyl5MnlnaBm0B5WMBMmmf-5hhh4-8xmOJ-vgJAQ?e=2Fl0MU)

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZypUOJgyl5MnlnaBm0B5WMBMmmf-5hhh4-8xmOJ-vgJAQ?e=2Fl0MU&download=1" -O PLASS/model/human_detect_tracker_checkpoint.pt
```

### Download All

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EepW2QK6ygJMpTSpk7zsiXoBo3NSq4KibNp-tAgOsCNRDA?e=EqOkKb&download=1" -O PLASS/model/selfharm_pose_checkpoint_best.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EaKk7gy9dq5AgyaBNK7gVcYByDjnz2mK7eQ0wyrPYdstug?e=1latwR&download=1" -O PLASS/model/human_pose_checkpoint.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ebk9j6XmHZNKkjTB95HXLOgBK856p99nlb2jMOeuKkeUYg?e=H3znvQ&download=1" -O PLASS/model/human_detect_checkpoint.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EZypUOJgyl5MnlnaBm0B5WMBMmmf-5hhh4-8xmOJ-vgJAQ?e=2Fl0MU&download=1" -O PLASS/model/human_detect_tracker_checkpoint.pt
```

## Linked Repository
This repository contains code for detecting and tracking people.<br/>
[Repository 🔗](https://github.com/qqaazz0222/PoliceLab_Plass_BoTSORT)
