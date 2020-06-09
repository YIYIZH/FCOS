#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/5/28 12:52 
 @Author : ZHANG 
 @File : count_number_scale.py 
 @Description:
"""
import json

#categories = {'head', 'person'}
splits = {'train', 'val'}

INF = 100000000

intervals = [
    [-1, 64],
    [64, 128],
    [128, 256],
    [256, 512],
    [512, INF]
]

min_size = 800
max_size = 1333


for split in splits:
    json_path = '/home/zhangyy/data/bdd/100k/bdd_coco_train_night+day_with_time.json'

    # count the number of boxes in each interval after rescaling
    counts = [0, 0, 0, 0, 0]
    with open(json_path, 'r') as obj:
        json_dict = json.load(obj)

        # get the resize ration for each image
        image_dict = json_dict['images']
        image_resize = dict()
        for image in image_dict:
            w = image['width']
            h = image['height']

            size = min_size

            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                ow = w
                oh = h

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            w_scale = float(ow) / w
            h_scale = float(oh) / h

            image_resize[image['id']] = [w_scale, h_scale]

        for anno in json_dict['annotations']:
            box = anno['bbox']

            # the width and height after resize
            w_box = box[2] * image_resize[anno['image_id']][0]
            h_box = box[3] * image_resize[anno['image_id']][1]

            for level in range(len(intervals)):
                if intervals[level][0] < max(w_box, h_box) < intervals[level][1]:
                    counts[level] += 1

    print(json_path)

    for level in range(len(intervals)):
        print('[{:d}, {:d}]: {:d}\n'.format(intervals[level][0], intervals[level][1], counts[level]))

    print('\nTotal number of boxes: {:d}'.format(sum(counts)))
