#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/4/10 19:17 
 @Author : ZHANG 
 @File : eval_per_class.py 
 @Description:
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = '/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_val_daytime.json'
results_json = '/home/yiyifrisky/code/FCOS/testing_dir/fcos_R_50_FPN_1x_bdd/inference/bdd_val_daytime/bbox.json'
cocoGt=COCO(annFile)
cocoDt=cocoGt.loadRes(results_json)

annType = 'bbox'

imgIds=sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, annType)
for cat in range(10):
    cocoEval.params.catIds = [cat+1] #person id : 1
    name = cocoGt.cats[cat+1]['name']
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(name)
    cocoEval.summarize()