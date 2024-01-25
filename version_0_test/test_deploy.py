# Python version: 3.11.7
#
# This file is for testing pickup_order
# 


import torch
import cv2
import sys
import pandas as pd
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', force_reload = False)

# will take a list of bounding boxes and return a list of indices
# indicating what order the robot should pick objects up in
def pickup_order (results) -> list:
    return None

# pick_n_frames is used to select n ideal frames to test pickup_order
# it selects a random sample of frames and picks n which have the most objects
# 
# since reading a frame can fail (cap.read()), you are not necessarily guarunteed
# n frames
# 
# also this takes me like 2 minutes to run on my laptop so yeah
#
# cap -> video capture from opencv (has to be opened)
# model -> YOLOv5 model
# samples -> how many random frames to pick from the video
# n -> number of frames to take from samples 
def pick_n_frames(cap: cv2.VideoCapture, model, samples: int = 100, n: int = 10) -> list:
    # number of frames in the video
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # pick random frames
    indices = np.random.choice(np.arange(nframes), samples, replace=False)

    # get a list of model outputs for each frame
    model_samples = []
    for idx in indices:
        # set the next frame to read to idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            results = model(frame).pandas().xyxy[0]
            model_samples.append(results)

    # return top n frames
    # sort by number of items seen
    return sorted(model_samples, key = lambda a: a.shape[0], reverse=True)[0:n]

def test_pickup_order():
    cap = cv2.VideoCapture("./far_west_test_video.mp4")

    test_frames = pick_n_frames(cap, model)

    cap.release()

    pickup_order(test_frames)

if __name__ == "__main__":
    test_pickup_order()