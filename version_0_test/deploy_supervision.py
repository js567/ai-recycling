# Python version: 3.11.6
#
# This file is for testing counting objects
# 
import torch
import supervision as sv
import numpy as np
import matplotlib.path as MPLP
import os
import cv2
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

SKIP_FRAMES = 2

# gets the bounding boxes of items that are higher than a set confidence
# input:
# predictions -> result of model inference
#
# output:
# all bounding box coords in the current frame, all classes associated with those bounding boxes
def get_bounding_boxes(predictions):
    df = predictions.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.3]

    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int), list(df["name"])

# gets the center of a given bounding box 
# input:
# bbox -> coords of a bounding box
#
# output:
# center of the bounding box
def get_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

# returns whether the center of the bounding box is in a defined rectangular area 
# input:
# xc -> x coord of center of bounding box
# yx -> y coord of center of bounding box
# shape -> coordinates of a rectangle area on the screen
#
# output:
# true/false whether the center of the bounding box is inside of the shape during the current frame
def is_in_area(xc, yc, shape):
    return MPLP.Path(shape).contains_point((xc, yc))

def main():
    print(f'Supervision version: {sv.__version__}')
    if sv.__version__ < '0.18.0':
        raise ValueError('Supervision version must be at least 0.18.0')
    
    weights = 'C:/Users/keywo/OneDrive/Desktop/Capstone/ai-recycling/version_0_test/best.pt' # Update this to the path of the best.pt file
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    print("Model loaded")

    gen = sv.get_video_frames_generator(source_path='C:/Users/keywo/OneDrive/Desktop/Capstone/ai-recycling/version_0_test/far_west_test_video.mp4')
    
    tracker = sv.ByteTrack() # TODO: figure out video framerate and pass it to the tracker
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=3)

    consecutive_frames_threshold = 10

    # smoother = sv.DetectionsSmoother()

    # Class counting.
    # We'll put seen IDs in a set, and count the number of unique IDs for any given class.
    # TODO: Change to a region/line based counting system. Supervision has ways to do this.
    seen_ids = [] # We'll limit this size to 30, and remove the oldest IDs when it gets too big
    class_counts = {}
    consecutive_frames = defaultdict(int)

    count = 0
    while True:
        count += 1
        frame = next(gen, None)

        if frame is None:
            break

        if count % SKIP_FRAMES != 0:
            continue

        results = model(frame)
        detections = sv.Detections.from_yolov5(results)
        detections = tracker.update_with_detections(detections)
        # detections = smoother.update_with_detections(detections)

        # Counting
        try:
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                if tracker_id not in seen_ids:
                    consecutive_frames[tracker_id] += 1
                    if consecutive_frames[tracker_id] >= consecutive_frames_threshold:
                        seen_ids.append(tracker_id)
                        seen_ids = seen_ids[-30:]
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                # Delete IDs that are not tracked consistently
                if tracker_id not in detections.tracker_id:
                    consecutive_frames.pop(tracker_id, None)
        except TypeError as e:
            # Sometimes the detections are empty, and zip doesn't like that
            pass

        # Display the counts of items on the screen
        for class_id, count in class_counts.items():
            cv2.putText(img=frame, text=f'{model.names[class_id]}: {count}', org=(200, (class_id+1)*150), fontFace=2, fontScale=3, color=(255,255,0), thickness=3)

        labels = [
            f'{tracker_id}: {model.names[class_id]}'
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(scene = frame.copy(), detections= detections)
        label_annotator.annotate(
            annotated_frame,
            detections=detections,
            labels=labels
        )
        resized_frame = cv2.resize(np.squeeze(annotated_frame), (1920, 1080))
        cv2.imshow('Test Video', resized_frame)

        if cv2.waitKey(5) == ord('q'):
            break

    # Print the counts, converting the class IDs to class names
    for class_id, count in class_counts.items():
        print(f'{model.names[class_id]}: {count}')
    

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
