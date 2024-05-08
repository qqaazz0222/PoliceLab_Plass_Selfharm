# Dongguk Police Lab Selfharm Module

## Installation

```
git clone https://github.com/qqaazz0222/PoliceLab_Plass_Selfharm.git
cd PoliceLab_Plass_Selfharm
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Donwload Checkpoint Files

### Selfharm Detect Checkpoint

File : configs/posec3d/slowonly_r50_selfharm/best.pth
Link : https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EblcEBmVyhpBp88RwcxGIoQBDVXdmfeyCU06Du7e79muCQ?e=jP0YVK

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EblcEBmVyhpBp88RwcxGIoQBDVXdmfeyCU06Du7e79muCQ?e=jP0YVK&download=1" -O configs/posec3d/slowonly_r50_selfharm/best.pth
```

### Human Pose Estimation Checkpoint

File : data/pretrain/human_pose_checkpoint.pth
Link : https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EYDBvXKs4_JOogMYEgEgjmsBk3zL_n8KqQeCFbLFpIzveg?e=Qybvqg

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EYDBvXKs4_JOogMYEgEgjmsBk3zL_n8KqQeCFbLFpIzveg?e=Qybvqg&download=1" -O data/pretrain/human_pose_checkpoint.pth
```

### Human Detect Estimation Checkpoint

File : data/pretrain/human_detect_checkpoint.pth
Link : https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EYZ6d-s6sDBAjS03amyczwQBJVS00dUayF5omzwYF7i-Nw?e=sgq0sz

```
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EYZ6d-s6sDBAjS03amyczwQBJVS00dUayF5omzwYF7i-Nw?e=sgq0sz&download=1" -O data/pretrain/human_detect_checkpoint.pth
```
