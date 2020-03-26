#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/3/19 15:49 
 @Author : ZHANG 
 @File : analyze_coco.py 
 @Description:
"""
from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt

#cats = coco.loadCats(coco.getCatIds())

coco = COCO('/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_bb.json')
input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_bb.json"
output = "/home/yiyifrisky/data/COCO2017/annotations/"

f = open(input)
f_data = json.load(f)
image = f_data['images']
c_0,c_1,c_2,c_3 = 0,0,0,0,
count,y_all = 0,0
for img in image:
    id = img['id']
    w = img['width']
    h = img['height']
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    leng = len(anns)
    for ann in anns:
        if ann['area'] == 0:
            print('area = 0 in ' + str(ann['image_id']))
            continue
        y = ann['bbox'][1] + ann['bbox'][3]
        if y/h < 1/4:
            c_0 +=1
        elif y/h <1/2:
            c_1 +=1
        elif y/h < 3/4:
            c_2 +=1
        else:
            c_3 +=1
        y_all += y/h
    count += leng

ave = y_all/count
c_all = c_0+c_1+c_2+c_3
print(ave,c_0/c_all,c_1/c_all,c_2/c_all,c_3/c_all,c_all)