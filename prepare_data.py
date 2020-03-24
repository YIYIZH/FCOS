#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/3/23 17:44 
 @Author : ZHANG 
 @File : prepare_data.py 
 @Description:
"""
import json
input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017.json"

f = open(input)
f_data = json.load(f)
bbox = f_data['annotations']

for box in bbox:
        if box['image_id'] == 200365:
            print(box['bbox'])
            print(box['category_id'])