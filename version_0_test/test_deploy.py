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
                # set center of each box
                box_a['centroid_x'] = (box_a['xmin'] + box_a['xmax'])/2
                box_a['centroid_y'] = (box_a['ymin'] + box_a['ymax'])/2
                box_b['centroid_x'] = (box_b['xmin'] + box_b['xmax'])/2
                box_b['centroid_y'] = (box_b['ymin'] + box_b['ymax'])/2
                
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

    # # calculate centroids
    # boxes['centroid_x'] = (boxes['xmin'] + boxes['xmax'])/2
    # boxes['centroid_y'] = (boxes['ymin'] + boxes['ymax'])/2

    # # figure out which item we're picking up
    # # then remove it
    # indices_to_remove = get_occlusions(boxes)
    # boxes = boxes.drop(indices_to_remove)

    # # sort indices by which one has the largest y value
    # # this should give us the items that are the closest to the end of the line
    # return boxes.sort_values(by=['centroid_y'], ascending=True)

    # figure out which ones are overlapping
    indices_to_remove = get_occlusions(boxes)

    # drop those rows from the data frame
    items_to_pickup = boxes.drop(indices_to_remove)

    # calculate centroids
    items_to_pickup['centroid_x'] = (items_to_pickup['xmin'] + items_to_pickup['xmax'])/2
    items_to_pickup['centroid_y'] = (items_to_pickup['ymin'] + items_to_pickup['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return items_to_pickup.sort_values(by=['centroid_y'], ascending=True)

# def swap_rows(df, row1, row2):
#     "Swaps two rows in a dataframe. Returns the dataframe."
#     df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
#     return df

def swap_occlusions (boxes: pd.DataFrame) -> None:
    """
    Swaps the order of the items that are overlapping.
    Overlaps with far enough distance are ignored. 
    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)
    frame: the opencv frame object
    @param results: a list of bounding_boxes (pandas Dataframe). Each row is a box.
    """
    box_columns = ['xmin', 'ymin', 'xmax', 'ymax']

    for i, box_a in boxes.iterrows():
        for j, box_b in boxes.iterrows():
            # swap boxes if conditions are met
            # if j is less than or equal to i, they've already been compared
            if j <= i:
                continue

            # get just the [xmin, ymin, xmax, ymax] part of each box
            bi, bj = box_a[box_columns].to_numpy(), box_b[box_columns].to_numpy()

            if intersect(bi, bj):
                # set center of each box
                box_a['centroid_x'] = (box_a['xmin'] + box_a['xmax'])/2
                box_a['centroid_y'] = (box_a['ymin'] + box_a['ymax'])/2
                box_b['centroid_x'] = (box_b['xmin'] + box_b['xmax'])/2
                box_b['centroid_y'] = (box_b['ymin'] + box_b['ymax'])/2

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
                
                # if the confidence of the first box is greater than the second, swap them
                if box_a['confidence'] > box_b['confidence']:
                    boxes.iloc[i], boxes.iloc[j] =  boxes.iloc[j].copy(), boxes.iloc[i].copy()
                else:
                    boxes.iloc[j], boxes.iloc[i] =  boxes.iloc[i].copy(), boxes.iloc[j].copy()

def pickup_order_v2 (bounding_boxes: pd.DataFrame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.
    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)
    frame: opencv frame the bounding boxes are from
    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    boxes = bounding_boxes.copy()

    # calculate centroids
    boxes['centroid_x'] = (boxes['xmin'] + boxes['xmax'])/2
    boxes['centroid_y'] = (boxes['ymin'] + boxes['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    sorted_boxes = boxes.sort_values(by=['centroid_y'], ascending=True)

    # swap the order of the items that are overlapping
    swap_occlusions(sorted_boxes)

    return sorted_boxes

def get_occlusions_v2 (bounding_boxes: pd.DataFrame) -> list:
    """
    Returns which objects are overlapping by iterating through columns and rows and drop the lower confidence.
    Overlaps with far enough distance are ignored. 
    Return type is a list of indices

    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)

    frame: the opencv frame object

    @param results: a list of bounding_boxes (pandas Dataframe). Each row is a box.
    """
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
                    # set center of each box
                    row['centroid_x'] = (row['xmin'] + row['xmax'])/2
                    row['centroid_y'] = (row['ymin'] + row['ymax'])/2
                    row2['centroid_x'] = (row2['xmin'] + row2['xmax'])/2
                    row2['centroid_y'] = (row2['ymin'] + row2['ymax'])/2

                    # set diagonal of each box
                    row['diagonal'] = np.sqrt((row['xmax'] - row['xmin'])**2 + (row['ymax'] - row['ymin'])**2)
                    row2['diagonal'] = np.sqrt((row2['xmax'] - row2['xmin'])**2 + (row2['ymax'] - row2['ymin'])**2)

                    # calculate size of each box
                    row['size'] = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
                    row2['size'] = (row2['xmax'] - row2['xmin']) * (row2['ymax'] - row2['ymin'])

                    # calculate size difference ratio
                    size_diff_ratio = abs(row['size'] - row2['size']) / max(row['size'], row2['size'])

                    # size threshold, for example, 0.5 means the size difference should not exceed 50%
                    size_threshold = 0.5

                    # sum of diagonal of each box, and take 1/8 of the sum, use it as the distance threshold
                    dist_threshold = (row['diagonal'] + row2['diagonal'])/8

                    # if the euclidean distance between the centroids is greater than the threshold, ignore
                    # if the size difference ratio is smaller than the threshold, ignore
                    if (np.sqrt((row['centroid_x'] - row2['centroid_x'])**2 + (row['centroid_y'] - row2['centroid_y'])**2) > dist_threshold) and (size_diff_ratio < size_threshold):
                        continue
                    
                    # if the confidence is higher, remove the other one
                    if row['confidence'] > row2['confidence']:
                        indices_to_remove.append(j)
                    else:
                        indices_to_remove.append(i)

    return indices_to_remove

def pickup_order_v3 (bounding_boxes: pd.DataFrame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)

    frame: opencv frame the bounding boxes are from

    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    # figure out which ones are overlapping
    indices_to_remove = get_occlusions_v2(bounding_boxes)

    # drop those rows from the data frame
    items_to_pickup = bounding_boxes.drop(indices_to_remove)

    # calculate centroids
    items_to_pickup['centroid_x'] = (items_to_pickup['xmin'] + items_to_pickup['xmax'])/2
    items_to_pickup['centroid_y'] = (items_to_pickup['ymin'] + items_to_pickup['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return items_to_pickup.sort_values(by=['centroid_y'], ascending=True)