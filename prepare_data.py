#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/3/23 17:44 
 @Author : ZHANG 
 @File : prepare_data.py 
 @Description:
"""

import json
from shutil import copyfile
from PIL import Image
import os

import concurrent.futures

width = 1280
height = 720
infile = '/dlwsdata3/public/bdd/day_night_256*256/test_latest/images/'
outfile = '/dlwsdata3/public/bdd/day_night_256*256/test_latest/test1/'


def resize_fake(files):
    im = Image.open(infile + files)
    out = im.resize((width, height), Image.ANTIALIAS)
    out.save(outfile + files)


if __name__ == '__main__':
    dirs = os.listdir(infile)
    files = [file for file in dirs if 'fake' not in file]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(resize_fake, files)



input = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/bdd100k_labels_images_val.json"
str = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/val/"
dst_day = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/val_daytime/"
dst_night = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/val_night/"
dst_dusk = "/dlwsdata3/yiyifrisky/bdd/bdd100k/images/100k/dusk/"
def seperate_day_night():
    f = open(input)
    f_data = json.load(f)
    i, j = 0, 0
    for img in f_data:
        name = img['name']
        att = img['attributes']
        if att['timeofday'] == 'daytime':
            i += 1
            copyfile(str + name, dst_day + name)
        elif att['timeofday'] == 'night':
            j += 1
            copyfile(str + name, dst_night + name)
    print(i, j)
