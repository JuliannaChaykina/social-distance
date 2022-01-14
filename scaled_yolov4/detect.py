import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from scaled_yolov4.experimental import attempt_load
from scaled_yolov4.datasets import LoadStreams, LoadImages, letterbox
from scaled_yolov4.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from scaled_yolov4.torch_utils import select_device, load_classifier, time_synchronized


device = None
model = None
names = None
imgsz = None

def detect(img):
    global device, model, names, imgsz
    orig_shape = img.shape
    with torch.no_grad():
        # Initialize
        if device is None:
            device = select_device("0")

        # Load model
        if model is None:
            model = attempt_load('scaled_yolov4/yolov4-p5.pt', map_location=device)  # load FP32 model
            names = model.module.names if hasattr(model, 'module') else model.names
            imgsz = check_img_size(416, s=model.stride.max())  # check img_size
            img_tmp = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img_tmp)  # run once
     
        # Run inference
        img = letterbox(img, new_shape=imgsz)[0]
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        curr_shape = img.shape

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred)

        objs = []
        for detection in pred:
            detection[:, :4] = scale_coords(curr_shape[2:], detection[:, :4], orig_shape).round()
            for *xyxy, conf, id in detection:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                x = xywh[0]
                y = xywh[1]
                w = xywh[2]/2
                h = xywh[3]/2

                obj = {}
                obj['xmin'] = int(x-w)
                obj['ymin'] = int(y-h)            
                obj['xmax'] = int(x+w)
                obj['ymax'] = int(y+h)
                obj['confidence'] = float(conf)
                obj['name'] = names[int(id)]
                objs.append(obj)
        return objs

