#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:17:33 2020

@author: kthiruko
"""
import cv2
import argparse
import os
import numpy as np
import platform
import shutil
import time
from pathlib import Path
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import PIL
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, ImgTransform

from utils.utils import *
from utils.torch_utils import select_device, load_classifier, time_synchronized


app = Flask(__name__)

# ## parameters in dict()
# default_params = {"conf_thres" : 0.25,
# "iou_thresh" : 0.5,
# "image_size" : 416,
# "weights" : "weights/yolov5s.pt",
# "source" : "/",
# "view_img" : False,
# "save_txt" : True,
# "text_path" : "result/text/"
# "device" : None ,
# "output_name" : ""
# }

conf_thres=0.25
iou_thres=0.5
image_size=416
weights="weights/yolov5s.pt"
source="/"
view_img=False
save_txt=True
text_path="result/text/"
device="cpu" 
output_name=""
save_img = True
img_path = "result/img/"



# load in weights and classes
# Initialize
# set_logging()
device = select_device(device)

if not os.path.exists(text_path):
    # shutil.rmtree(text_path)  # delete output folder
    os.makedirs(text_path)  # make new output folder

# Load model
model = attempt_load(weights)  # load FP32 model
image_size = check_img_size(image_size, s=model.stride.max())  # check img_size
print('weights loaded')

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
print('classes loaded')

# # Initialize Flask application
# app = Flask(__name__)


# route http posts to this method
@app.route('/test')
def hello():
    return 'Hello, World!'


# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = np.array(PIL.Image.open(image))

    # create list of responses for current image
    responses = []
    detection_text = []

    img = ImgTransform(img_raw, img_size=image_size)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s, im0 =  '', img_raw
        s += '%gx%g ' % img.shape[2:]  # print string
        
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    detection = ('%g, %g, %g, %g, %s, %.2f') % (*xywh, names[int(cls)], conf)
                    detection_text.append(detection)
                    

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            
                responses.append({
                    "class": names[int(cls)],
                    "confidence": float("{0:.2f}".format(conf))
                })    
    
    if save_img:
        cv2.imwrite(img_path+image_name+".jpg", im0)            
                            
    det_response = {
        "image": image_name,
        "detections": responses
        }
    
    # write the text detection to the file
    with open(text_path + image_name.split('.')[0] + '.txt', 'w') as f:
        for item in detection_text:
            f.write("%s\n" % item)  # label format
    
    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (s, t2 - t1))

    #remove temporary images 
    os.remove(image_name)
    
    try:
        return jsonify({"response":det_response}), 200
    except FileNotFoundError:
        abort(404)


# API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = np.array(PIL.Image.open(image))

    # create list of responses for current image
    responses = []

    img = ImgTransform(img_raw, img_size=image_size)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s, im0 =  '', img_raw
        s += '%gx%g ' % img.shape[2:]  # print string
        
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):  
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
    
    response = img_encoded.tostring()
    
    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (s, t2 - t1))

    #remove temporary images
    os.remove(image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)

