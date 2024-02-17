# Python version: 3.11.7
#
# This file is for testing pickup_order
# 

import pandas as pd
import numpy as np

def intersect (b1: np.array, b2: np.array) -> bool:
    """
    Takes two bounding boxes and determines if there's an intersection

    input format: [min_x, min_y, max_x, max_y]
    """

    x_overlap = (b1[0] <= b2[0] <= b1[2]) or (b1[0] <= b2[2] <= b1[2])
    y_overlap = (b1[1] <= b2[1] <= b1[3]) or (b1[1] <= b2[3] <= b1[3])

    return x_overlap and y_overlap

def get_occlusions (boxes: pd.DataFrame) -> list:
    """
    Detects occlusions and returns a list of indices for which objects should be ignored.

    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)
    """
    
    box_columns = ['xmin', 'ymin', 'xmax', 'ymax']
    indices_to_remove = []

    for i, box_a in boxes.iterrows():
        for j, box_b in boxes.iterrows():
            # ignore boxes we don't need to calculate
            # if j is less than or equal to i, they've already been compared
            if j <= i:
                continue

            # get just the [xmin, ymin, xmax, ymax] part of each box
            bi, bj = box_a[box_columns].to_numpy(), box_b[box_columns].to_numpy()

            if intersect(bi, bj):
                # get diagonal of each box
                diagonal_a = np.sqrt((box_a['xmax'] - box_a['xmin'])**2 + (box_a['ymax'] - box_a['ymin'])**2)
                diagonal_b = np.sqrt((box_b['xmax'] - box_b['xmin'])**2 + (box_b['ymax'] - box_b['ymin'])**2)

                # calculate area of each box
                area_a = (box_a['xmax'] - box_a['xmin']) * (box_a['ymax'] - box_a['ymin'])
                area_b = (box_b['xmax'] - box_b['xmin']) * (box_b['ymax'] - box_b['ymin'])

                # calculate area difference ratio
                size_diff_ratio = abs(area_a - area_b) / max(area_a, area_b)

                # size threshold, for example, 0.5 means the size difference should not exceed 50%
                size_threshold = 0.5

                # sum of diagonal of each box, and take 1/8 of the sum, use it as the distance threshold
                dist_threshold = (diagonal_a + diagonal_b)/8

                # if the euclidean distance between the centroids is greater than the threshold, ignore
                # if the size difference ratio is smaller than the threshold, ignore
                centroid_distance = np.sqrt((box_a['centroid_x'] - box_b['centroid_x'])**2 + (box_a['centroid_y'] - box_b['centroid_y'])**2)
                if (centroid_distance > dist_threshold) and (size_diff_ratio < size_threshold):
                    continue
                
                # ignore the item with lower confidence
                if box_a['confidence'] > box_b['confidence']:
                    indices_to_remove.append(j)
                else:
                    indices_to_remove.append(i)
    
    return indices_to_remove
    
def pickup_order (boxes: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the optimal order to pick up objects in. Returns a sorted version of bounding boxes passed in.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)
    """

    # calculate centroids
    boxes['centroid_x'] = (boxes['xmin'] + boxes['xmax'])/2
    boxes['centroid_y'] = (boxes['ymin'] + boxes['ymax'])/2

    # figure out which item we're picking up
    # then remove it
    indices_to_remove = get_occlusions(boxes)
    boxes = boxes.drop(indices_to_remove)

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return boxes.sort_values(by=['centroid_y'], ascending=True)
