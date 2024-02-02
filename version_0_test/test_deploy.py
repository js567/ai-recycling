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

def intersect (b1: np.array, b2: np.array) -> bool:
    """
    Takes two bounding boxes and determines if there's an intersection

    input format: [min_x, min_y, max_x, max_y]

    """

    # b1[0] = left side, b1[2] = right side
    # b1[1] = bottom, b1[3] = top
    x_overlap = (b1[0] <= b2[0] <= b1[2]) or (b1[0] <= b2[2] <= b1[2])
    y_overlap = (b1[1] <= b2[1] <= b1[3]) or (b1[1] <= b2[3] <= b1[3])

    return x_overlap and y_overlap

def calculate_background_percentage(b: np.array, frame) -> np.float32:
    """
    Takes a bounding box and a frame and returns what percent of the bounding box
    is the background (assuming background is black)

    returns a float between 0 and 1
    """
    return 0

def get_occlusions (bounding_boxes: pd.DataFrame, frame) -> list:
    """Returns which objects are overlapping. Return type is a list of indices

    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)

    frame: the opencv frame object

    @param results: a list of bounding_boxes (pandas Dataframe). Each row is a box.
    """

    # list of indices to remove
    indices_to_remove = []

    box_columns = ['xmin', 'ymin', 'xmax', 'ymax']

    # iterate through each row
    for i, row in bounding_boxes.iterrows():
        # iterate through each row again
        for j, row2 in bounding_boxes.iterrows():
            # don't compare to self
            # also, if j < i, then both of these rectangles have already been compared
            # otherwise we do 2 times the number of comparisons
            if i == j or j < i:
                continue

            # get just the [xmin, ymin, xmax, ymax] part of each row
            bi, bj = row[box_columns].to_numpy(), row2[box_columns].to_numpy()

            # if they intersect, choose the one with less background showing
            if intersect(bi, bj):
                bi_background = calculate_background_percentage(bi, frame)
                bj_background = calculate_background_percentage(bj, frame)

                indices_to_remove.append(i) if bi_background > bj_background else indices_to_remove.append(j)



    return indices_to_remove

def pickup_order (bounding_boxes: pd.DataFrame, frame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)

    frame: opencv frame the bounding boxes are from

    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    # figure out which ones are overlapping
    indices_to_remove = get_occlusions(bounding_boxes, frame)

    # drop those rows from the data frame
    items_to_pickup = bounding_boxes.drop(indices_to_remove)

    # calculate centroids
    items_to_pickup['centroid_x'] = (items_to_pickup['xmin'] + items_to_pickup['xmax'])/2
    items_to_pickup['centroid_y'] = (items_to_pickup['ymin'] + items_to_pickup['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return items_to_pickup.sort_values(by=['centroid_y'], ascending=True)

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