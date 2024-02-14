# Python version: 3.11.7
#
# This file is for testing pickup_order
# 

import cv2
import pandas as pd
import numpy as np

BLACK_PIXEL = np.zeros(3)
WHITE_PIXEL = np.array([255,255,255])

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

def transform_pixel(px: np.array) -> np.array:
    """
    Helper function for calculate_background_percentage

    @param px: numpy array with a pixel's rgb value

    outputs a white pixel or a black pixel depending on which one px is closer to
    """

    whitedist = np.linalg.norm(px-WHITE_PIXEL)
    blackdist = np.linalg.norm(px-BLACK_PIXEL)

    return WHITE_PIXEL if whitedist < blackdist else BLACK_PIXEL

def calculate_background_percentage(b: np.array, frame) -> np.float32:
    """
    Takes a bounding box and a frame and returns what percent of the bounding box
    is the background (assuming background is black)

    NOTE: Currently needs more testing.

    returns a float between 0 and 1
    """
    # get position of box
    xmin, xmax, ymin, ymax = int(b[0]), int(b[2]), int(b[1]), int(b[3])

    # get just the bounding box pixels
    region = frame[xmin:xmax, ymin:ymax]

    # ¯\_(ツ)_/¯
    if region.size == 0:
        return 1 

    # this is better at calculating backgrounds but is slow as shit
    '''
    # make every pixel either black or white
    transformed_region = np.apply_along_axis(transform_pixel, 2, region)

    # calculate the percentage of black pixels
    return np.sum(transformed_region == WHITE_PIXEL)/region.size

    '''
    
    # upper bound for each r, g, and b value
    upper_bound = np.array([90,90,90])

    # element-wise compare each pixel's rgb values
    # to upper_bound
    pixel_comparisons = region < upper_bound

    # calculate how many pixels are r < 50, g < 50 and b < 50
    # it does this by summing the comparisons from earlier (True, True, False) = 2
    # then comparing those sums to 3
    background_n = np.sum(np.sum(pixel_comparisons, axis=2) == 3)

    # ALTERNATE CODE:
    # the comparison earlier means that [1,1,2] < [2,2,2] is false
    # but we would probably think that [1,1,1] is the background anyway
    # this method uses the length of the rgb vector as a comparison instead
    # however, this doesn't take into account the "direction" of the rgb so it would
    # include something like [120, 1, 1] or something when it shouldn't
    # currently it only gives a little bit more than the other method
    #
    # so.... the best way might be to take the dot product of each pixel and [50,50,50]?
    # might look into that in the future

        # upper bound for what pixel rgb values we
        # will want to capture as background
        #upper_bound = np.linalg.norm(np.array([50,50,50]))

        #rgb_norms = np.linalg.norm(region, axis=2)

        #background_n = np.sum(rgb_norms < upper_bound)

    # return the percentage of the bounding box the background takes up
    return background_n / region.size

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

def get_occlusions_v2 (bounding_boxes: pd.DataFrame, frame) -> list:
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

def swap_rows(df, row1, row2):
    "Swaps two rows in a dataframe. Returns the dataframe."
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df

def swap_occlusions (bounding_boxes: pd.DataFrame, frame) -> None:
    """
    Swaps the order of the items that are overlapping.
    Overlaps with far enough distance are ignored. 

    Columns for the DataFrame input: xmin, ymin, xmax, ymax, confidence, class, name
                                     float64, float64, float64, float64, float64, int64, object (string)

    frame: the opencv frame object

    @param results: a list of bounding_boxes (pandas Dataframe). Each row is a box.
    """
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

                    # if the confidence is higher, swap the two rows
                    if row['confidence'] < row2['confidence']:
                        bounding_boxes = swap_rows(bounding_boxes, i, j)
    
def pickup_order (bounding_boxes: pd.DataFrame, frame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)

    frame: opencv frame the bounding boxes are from

    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    # figure out which ones are overlapping
    indices_to_remove = get_occlusions_v2(bounding_boxes, frame)

    # drop those rows from the data frame
    items_to_pickup = bounding_boxes.drop(indices_to_remove)

    # calculate centroids
    items_to_pickup['centroid_x'] = (items_to_pickup['xmin'] + items_to_pickup['xmax'])/2
    items_to_pickup['centroid_y'] = (items_to_pickup['ymin'] + items_to_pickup['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    return items_to_pickup.sort_values(by=['centroid_y'], ascending=True)

def pickup_order_v2 (bounding_boxes: pd.DataFrame, frame) -> pd.DataFrame:
    """Determines the optimal order to pick up objects in. Returns a sorted version of bounding_boxes.

    Columns for the dataframe: xmin, ymin, xmax, ymax, confidence, class, name
                               float64, float64, float64, float64, float64, int64, object (string)

    frame: opencv frame the bounding boxes are from

    @param results: a list of bounding boxes (as a pandas dataframe)
    """

    items_to_pickup = bounding_boxes.copy()

    # calculate centroids
    items_to_pickup['centroid_x'] = (items_to_pickup['xmin'] + items_to_pickup['xmax'])/2
    items_to_pickup['centroid_y'] = (items_to_pickup['ymin'] + items_to_pickup['ymax'])/2

    # sort indices by which one has the largest y value
    # this should give us the items that are the closest to the end of the line
    items_to_pickup.sort_values(by=['centroid_y'], ascending=True)

    # swap the order of the items that are overlapping
    swap_occlusions(items_to_pickup, frame)

    return items_to_pickup

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