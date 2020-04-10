#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/4/9 16:50 
 @Author : ZHANG 
 @File : bdd_7k_coco.py 
 @Description:
"""
import os
import json
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument(
          "-l", "--label_dir",
          default="/dlwsdata3/yiyifrisky/bdd/bdd100k/labels/100k/train/",
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default="/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/",
          help="save root directory of BDD label Json files",
    )
    return parser.parse_args()


def bdd2coco_detection(id_dict, fn):
    files = os.listdir(args.label_dir)
    counter = 0
    images = list()
    annotations = list()
    count = -1
    for file in files:
        p = open(args.label_dir + file)
        train_labels = json.load(p)

        counter += 1
        image = dict()
        image['file_name'] = train_labels['name'] + '.jpg'
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter
        empty_image = True

        for l in train_labels['frames'][0]['objects']:
            annotation = dict()
            if l['category'] in id_dict.keys():
                count += 1
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = l['box2d']['x1']
                y1 = l['box2d']['y1']
                x2 = l['box2d']['x2']
                y2 = l['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[l['category']]
                annotation['ignore'] = 0
                annotation['id'] = count
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue
        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)


if __name__ == '__main__':

    args = parse_arguments()

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bike"},
        {"supercategory": "none", "id": 7, "name": "motor"},
        {"supercategory": "none", "id": 8, "name": "traffic light"},
        {"supercategory": "none", "id": 9, "name": "traffic sign"},
        {"supercategory": "none", "id": 10, "name": "train"}
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create BDD training set detections in COCO format
    print('Converting training set...')

    out_fn = os.path.join(args.save_path,
                          'bdd100k_labels_images_det_coco_train.json')

    bdd2coco_detection(attr_id_dict, out_fn)