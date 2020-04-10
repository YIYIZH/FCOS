#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/4/9 15:38 
 @Author : ZHANG 
 @File : get_7K_label.py
 @Description:
"""
import json
from shutil import copyfile
from pycocotools.coco import COCO
import os
input_train = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_train.json"
input_val = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_val.json"
str = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/train/"
dst_day = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/daytime100/"
dst_night = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/night100/"
dst_dusk = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/dusk/"

l = []
path = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/train/"
for root, dirs, files in os.walk(path):
    l.append(files)

f = open(input_train)
f_data = json.load(f)

for img in f_data:
    name = img['name']
    l[0].remove(name)

print(l)




