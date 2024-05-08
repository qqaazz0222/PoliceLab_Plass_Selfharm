import os
from mmcv import load, dump
from pyskl.smp import *

path = "./selfharm"
list_path = os.path.join(path, "selfharm.list")
train_path = os.path.join(path, "train.json")
test_path = os.path.join(path, "test.json")
train = load(train_path)
test = load(test_path)
tmpl = './selfharm/rgb/{}.mp4'

lines = [(tmpl + ' {}').format(x['vid_name'], x['label']) for x in train + test]
mwlines(lines, list_path)