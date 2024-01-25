# Python version: 3.11.6
#
# This file is for testing counting objects
# 
import torch
# import cv2
import supervision as sv
import numpy as np
import matplotlib.path as MPLP
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

# def main():
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/keywo/OneDrive/Desktop/Capstone/ai-recycling/version_0_test/best.pt', force_reload=True)
#     print("Model loaded")
#     cap = cv2.VideoCapture('C:/Users/keywo/OneDrive/Desktop/Capstone/ai-recycling/version_0_test/far_west_test_video.mp4')
#     print("Video loaded")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     RECTANGLE = np.array([
#         [0, int(height * 3/4)], 
#         [width, int(height * 3/4)],
#         [width, int(height * 3/4 + 100)],
#         [0, int(height * 3/4 + 100)]
#     ])

#     classes = {}
#     print("Starting video")
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if not ret:
#             break

#         results = model(frame)

#         bounding_boxes, class_list = get_bounding_boxes(results)
        
#         # Count the items that are in the rectangle
#         for box, c in zip(bounding_boxes, class_list):
#             xc, yc = get_center(box)
#             if is_in_area(xc, yc, RECTANGLE):
#                 if c not in classes:
#                     classes[c] = 0
#                 classes[c] += 1

#         # Display the counts of items on the screen
#         for i, c in enumerate(classes):
#             cv2.putText(img=frame, text="{}: {}".format(c, classes[c]), org=(200, (i+1)*150), fontFace=2, fontScale=3, color=(255,255,0), thickness=3)

#         # Rectangle area for counting
#         cv2.polylines(img=frame, pts=[RECTANGLE], isClosed=True, color=(0,0,255), thickness=4)

#         # Display the video on across all monitors
#         im = ResizeWithAspectRatio(np.squeeze(results.render()), width=1920)
#         cv2.imshow('Test Video', im)
#         # cv2.imshow('Test Video', np.squeeze(frame))


#         if cv2.waitKey(5) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    print("Model loaded")
    track = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        detections = sv.Detections.from_yolov5(model(frame))
        detections = track.update_with_detections(detections)

        labels = [
            f'{tracker_id}: {model.names[class_id]}'
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]
        annotated_frame = box_annotator.annotate(
            frame.copy(),
            detections=detections)
        return label_annotator.annotate(
            annotated_frame,
            detections=detections,
            labels=labels
        )
    
    sv.process_video(
        source_path='far_west_test_video.mp4',
        callback=callback,
        target_path='output.mp4'
    )

if __name__ == '__main__':
    main()
