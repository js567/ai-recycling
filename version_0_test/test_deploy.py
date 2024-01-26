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

def get_occlusions (bounding_boxes: pd.DataFrame) -> list:
    """Returns which objects are overlapping. Return type is a list of indices

    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)

    @param results: a list of bounding_boxes (pandas Dataframe). Each row is a box.
    """
    
    # list of indices to remove
    indices_to_remove = []

    # iterate through each row
    for i, row in bounding_boxes.iterrows():
        # iterate through each row again
        for j, row2 in bounding_boxes.iterrows():
            # don't compare to self
            if i == j:
                continue

            # if the x values overlap
            if row['xmin'] <= row2['xmax'] and row['xmax'] >= row2['xmin']:
                # if the y values overlap
                if row['ymin'] <= row2['ymax'] and row['ymax'] >= row2['ymin']:
                    # if the confidence is higher, remove the other one
                    if row['confidence'] > row2['confidence']:
                        indices_to_remove.append(j)
                    else:
                        indices_to_remove.append(i)

    return indices_to_remove

def pickup_order (bounding_boxes: pd.DataFrame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)

    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    # figure out which ones are overlapping
    indices_to_remove = get_occlusions(bounding_boxes)

    # drop those rows from the data frame
    items_to_pickup = bounding_boxes.drop(indices_to_remove)

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return items_to_pickup.sort_values(by=['ymax'], ascending=False)

def pick_n_frames(cap: cv2.VideoCapture, model, samples: int = 50, n: int = 7) -> list:
    """Picks ideal frames from a video to test with. Tries to get a good number of objects.
    
    Works by picking random frames and sorting those by the number of objects.

    Returns a list of pandas Dataframes. Each item in the array is the list of bounding boxes
    identified by YOLOv5 for that frame.

    @param cap: the opened video capture from opencv
    @param model: the YOLOv5 model object
    @param samples: the number of random samples to take from the video
    @param n: how many frames to take out of samples
    """

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

    for i, bounding_boxes in enumerate(test_frames):
        test_frames[i] = pickup_order(bounding_boxes)
        print(test_frames[i])

if __name__ == "__main__":
    test_pickup_order()