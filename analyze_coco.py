#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/12/25 17:19 
 @Author : ZHANG 
 @File : analyze_coco.py 
 @Description:
"""

from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt

coco = COCO('/home/yiyifrisky/data/COCO2017/annotations/instances_train2017_ss.json')
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]

ins = "/home/yiyifrisky/data/COCO2017/annotations/analyze_ss.json"
inm = "/home/yiyifrisky/data/COCO2017/annotations/analyze_mm.json"
inb = "/home/yiyifrisky/data/COCO2017/annotations/analyze_bb.json"
output = "result.png"

fs = open(ins)
ss = json.load(fs)
fm = open(inm)
mm = json.load(fm)
fb = open(inb)
bb = json.load(fb)

x =list(range(len(ss)))
total_width, n = 0.9, 3
width = total_width / n

plt.bar(x, ss, width=width, label='s', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, mm, width=width, label='m', tick_label=nms, fc='r')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, bb, width=width, label='b', tick_label=nms, fc='b')

plt.legend()
fig = plt.gcf()
fig.set_size_inches(60, 8)
fig.savefig(output)
