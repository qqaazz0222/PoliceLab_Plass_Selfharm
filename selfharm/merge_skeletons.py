from mmcv import load, dump
from pyskl.smp import *
train = load('train.json')
test = load('test.json')
annotations = load('selfharm_annos.pkl')
split = dict()
split['train'] = [x['vid_name'] for x in train]
split['test'] = [x['vid_name'] for x in test]
dump(dict(split=split, annotations=annotations), 'selfharm_hrnet.pkl')