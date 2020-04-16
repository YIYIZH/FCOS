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
from pycocotools.cocoeval import COCOeval


def create_fake_json():
    f = open("/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train_daytime.json")
    coco = COCO("/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train_daytime.json")
    f_data = json.load(f)
    image = f_data['images']
    for img in image:
        img['file_name'] = img['file_name'].split('.')[0] + '_fake.png'

    with open("/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train_daytime_fake.json", "w") as jsonFile:
        json.dump(f_data, jsonFile)


create_fake_json()

input_train = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_train.json"
input_val = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_val.json"
str = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/train/"
dst_day = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/daytime100/"
dst_night = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/night100/"
dst_dusk = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/dusk/"


def count_ann_by_time_cat(input):
    f = open(input)
    f_data = json.load(f)
    count_day = dict.fromkeys(['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train'], 0)
    count_night = dict.fromkeys(['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train'], 0)

    for img in f_data:
        att = img['attributes']
        cat = img['labels']
        if att['timeofday'] == 'daytime':
            for li in cat:
                cat = li['category']
                if cat in count_day:
                    count_day[cat] += 1
        elif att['timeofday'] == 'night':
            for li in cat:
                cat = li['category']
                if cat in count_night:
                    count_night[cat] += 1

    print(count_day, count_night)


def count_img_by_time(input):
    f = open(input)
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


annFile = '/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_val.json'
results_json = '/home/yiyifrisky/code/FCOS/testing_dir/fcos_R_50_FPN_1x_bdd/inference/bdd_val/bbox.json'


def eval_per_class(annFille, results_json):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(results_json)

    annType = 'bbox'

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    for cat in range(10):
        cocoEval.params.catIds = [cat + 1]  # person id : 1
        name = cocoGt.cats[cat + 1]['name']
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        print(name)
        cocoEval.summarize()


def count_cat():
    f = open("/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train.json")
    coco = COCO("/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_det_coco_train.json")
    f_data = json.load(f)
    image = f_data['images']
    i = len(image)
    n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for img in image:
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] == 1:
                n_0 += 1
            elif ann['category_id'] == 2:
                n_1 += 1
            elif ann['category_id'] == 3:
                n_2 += 1
            elif ann['category_id'] == 4:
                n_3 += 1
            elif ann['category_id'] == 5:
                n_4 += 1
            elif ann['category_id'] == 6:
                n_5 += 1
            elif ann['category_id'] == 7:
                n_6 += 1
            elif ann['category_id'] == 8:
                n_7 += 1
            elif ann['category_id'] == 9:
                n_8 += 1
            elif ann['category_id'] == 10:
                n_9 += 1

    print(n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9)