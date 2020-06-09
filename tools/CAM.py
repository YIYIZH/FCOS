#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2020/6/4 11:33 
 @Author : ZHANG 
 @File : CAM.py 
 @Description:
"""

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import torch
from fcos_core.utils.comm import synchronize, get_rank
import argparse
from fcos_core.config import cfg
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.logger import setup_logger
# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
CATEGORIES_bdd = [
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train",
    ]

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnCAM_bdd(feature_conv):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = feature_conv.reshape((nc, h*w))
    cam = cam[2].reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="../configs/fcos/fcos_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")


    model = build_detection_model(cfg)
    #model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    model_id = 4
    if model_id == 1:
        net = models.squeezenet1_1(pretrained=True)
        finalconv_name = 'features' # this is the last conv layer of the network
    elif model_id == 2:
        net = models.resnet18(pretrained=True)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=True)
        finalconv_name = 'features'
    elif model_id == 4:
        net = model
        finalconv_name = 'cls_logits'

    net.eval()
    #net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    net._modules.get('rpn')._modules.get('head_shared')._modules.get('cls_logits').register_forward_hook(hook_feature)

    # get the softmax weight
    #params = list(net.parameters())
    #weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    response = requests.get(IMG_URL)
    img_pil = Image.open(io.BytesIO(response.content))
    img_pil.save('test.jpg')


    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    '''
    # download the imagenet category list
    classes = {int(key): value for (key, value)
               in requests.get(LABELS_URL).json().items()}
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    '''
    classes_bdd = {int(key): value for (key, value) in enumerate(CATEGORIES_bdd)}

    # generate class activation mapping for the top1 prediction
    #CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    CAMs = returnCAM_bdd(features_blobs[2])

    # render the CAM and output
    #print('output CAM.jpg for the top1 prediction: %s' % classes_bdd[idx[0]])
    img = cv2.imread('night.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('night2_2.jpg', result)


if __name__ == "__main__":
    main()