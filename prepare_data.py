#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/12/23 10:34 
 @Author : ZHANG 
 @File : prepare_data.py 
 @Description:
"""
import json
'''
###################
##################
###################
from pycocotools.coco import COCO

input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_ss.json"
output = "/home/yiyifrisky/data/COCO2017/annotations/"
f = open(input)
f_data = json.load(f)
s = open(output + 'analyze_ss.json', 'w')
bbox = f_data['annotations']
count = [0]*80

coco = COCO(input)
l = coco.getCatIds()
json_category_id_to_contiguous_id = {
    v: i + 1 for i, v in enumerate(l)
        }

for seg in bbox:
    id = seg['category_id']
    idx = json_category_id_to_contiguous_id[id]
    count[idx-1] += 1

s.write(json.dumps(count))

####################
###############
####################

import json
input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_b.json"
output = "/home/yiyifrisky/data/COCO2017/annotations/"
f = open(input)
f_data = json.load(f)
s = open(output + 'instances_train2017_bb.json', 'w')
s.write('{"info": ' + json.dumps(f_data['info']) + ',"licenses": ' + json.dumps(f_data['licenses']) + ',"images": '
            + json.dumps(f_data['images']) + ',"annotations": [')

bbox = f_data['annotations']
first_s = 1
for seg in bbox:
    if seg['iscrowd'] == 0:
        if first_s == 1:
            s.write(json.dumps(seg))
            first_s = 0
        else:
            s.write(',' + json.dumps(seg))
s.write('],"categories": ' + json.dumps(f_data['categories']) + '}')

###############
#################
##############

input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017.json"
output = "/home/yiyifrisky/data/COCO2017/annotations/"
metric = [32*32, 96*96]

f = open(input)
f_data = json.load(f)

s = open(output + 'instances_train2017_s.json', 'w')
s.write('{"info": ' + json.dumps(f_data['info']) + ',"licenses": ' + json.dumps(f_data['licenses']) + ',"images": '
            + json.dumps(f_data['images']) + ',"annotations": [')

m = open(output + 'instances_train2017_m.json', 'w')
m.write('{"info": ' + json.dumps(f_data['info']) + ',"licenses": ' + json.dumps(f_data['licenses']) + ',"images": '
            + json.dumps(f_data['images']) + ',"annotations": [')

b = open(output + 'instances_train2017_b.json', 'w')
b.write('{"info": ' + json.dumps(f_data['info']) + ',"licenses": ' + json.dumps(f_data['licenses']) + ',"images": '
            + json.dumps(f_data['images']) + ',"annotations": [')

bbox = f_data['annotations']
first_s = 1
first_m = 1
first_b = 1
for seg in bbox:
    if seg['area'] < metric[0]:
        if first_s == 1:
            s.write(json.dumps(seg))
            first_s = 0
        else:
            s.write(',' + json.dumps(seg))

    elif seg['area'] < metric[1]:
        if first_m == 1:
            m.write(json.dumps(seg))
            first_m = 0
        else:
            m.write(',' + json.dumps(seg))
    else:
        if first_b == 1:
            b.write(json.dumps(seg))
            first_b = 0
        else:
            b.write(',' + json.dumps(seg))


s.write('],"categories": ' + json.dumps(f_data['categories']) + '}')
m.write('],"categories": ' + json.dumps(f_data['categories']) + '}')
b.write('],"categories": ' + json.dumps(f_data['categories']) + '}')


######################## debug ####################################
############################
#############################
input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_s.json"
output = "/home/yiyifrisky/data/COCO2017/annotations/"

f = open(input)
f_data = json.load(f)
bbox = f_data['annotations']

list = [205116, 38490, 263428, 116339, 226658, 130215, 141257, 59309, 190441, 372067, 82511, 511892, 501116, 72961, 318736, 509397]
list1 = [180099, 558608, 512173, 447437, 185234, 573146, 536498, 305404, 525179, 105531, 492166, 354012,141382, 436492, 52147, 546191]
list2 = [166529, 381608, 99910, 312081]
list3 = [190441]
num = []
for i in range (4):
    n = 0
    for box in bbox:
        if box['image_id'] == list2[i] and box['iscrowd'] == 0:
            n +=1
            #print(box)
    num.append(n)
print(num)

'''
input = "/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_ss.json"

f = open(input)
f_data = json.load(f)
bbox = f_data['annotations']

list3 = [333775, 96414, 102377, 428440]
for i in range (4):
    print("1111")
    for box in bbox:
        if box['image_id'] == list3[i]:
            print(box['bbox'])
