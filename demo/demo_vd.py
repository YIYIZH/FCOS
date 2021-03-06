#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/3/20 20:36 
 @Author : ZHANG 
 @File : demo_vd.py 
 @Description:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from fcos_core.config import cfg
from predictor import COCODemo
from datasets import LoadImages
import torch_utils

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/fcos/fcos_R_50_FPN_1x_bdd.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="/home/yiyifrisky/code/FCOS/training_dir/bdd_nightwithfake/model_final.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--source',
        type=str,
        default='demo.avi',
        help='source'
    )  # input file/folder, 0 for webcam

    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_coco_classes = [
        0.4923645853996277, 0.4928510785102844, 0.5040897727012634,
        0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
        0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
        0.43564629554748535, 0.6089804172515869, 0.666087806224823,
        0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
        0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
        0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
        0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
        0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
        0.523737907409668, 0.47027698159217834, 0.5103300213813782,
        0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
        0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
        0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
        0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
        0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
        0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
        0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
        0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
        0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
        0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
        0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
        0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
        0.5164387822151184, 0.540651798248291, 0.5323763489723206,
        0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
        0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
        0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
        0.4224434792995453, 0.4998510777950287
    ]
    thresholds_for_bdd_classes = [0.48780617117881775, 0.5085304975509644, 0.5058003664016724,
                                  0.5046775341033936, 0.5318371057510376, 0.459444135427475,
                                  0.515097975730896, 0.4111378788948059, 0.4400688707828522,
                                  0.35690930485725403]

    #thresholds_for_classes = [0.5] * 80
    vid_path, vid_writer = None, None
    source = args.source
    dataset = LoadImages(source)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_bdd_classes,
        min_image_size=args.min_image_size
    )

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        save_path = "./out" + path
        t1 = torch_utils.time_synchronized()
        composite = coco_demo.run_on_opencv_image(im0s)
        t2 = torch_utils.time_synchronized()
        print('Done. (%.3fs)' % (t2 - t1))

        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*args.fourcc), fps, (w, h))
        vid_writer.write(composite)
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == "__main__":
    main()


