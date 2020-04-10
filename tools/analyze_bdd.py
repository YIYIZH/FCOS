#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/4/9 13:40 
 @Author : ZHANG 
 @File : analyze_bdd.py 
 @Description:
"""

import json
from shutil import copyfile
from pycocotools.coco import COCO

input_train = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train.json"
input_val = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_val.json"
str = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/train/"
dst_day = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/daytime100/"
dst_night = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/night100/"
dst_dusk = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/dusk/"

f = open(input_train)
f_data = json.load(f)
i, j, k, l = 0, 0, 0, 0
for img in f_data:
    att = img['attributes']
    if att['timeofday'] == 'daytime':
        i += 1
    elif att['timeofday'] == 'night':
        j += 1
    elif att['timeofday'] == 'dawn/dusk':
        k += 1
    else:
        print(att['timeofday'])
        l +=1
print(i, j, k, l)

