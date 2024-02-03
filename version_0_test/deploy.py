import torch
import cv2
import numpy as np
import test_deploy as t_pickup_order
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', force_reload=True)

cap = cv2.VideoCapture('far_west_test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame)
    rendered_output = np.squeeze(results.render())
    ordered_boxes = t_pickup_order.pickup_order(results.pandas().xyxy[0], frame)

    # label each item based on its priority
    # we can't just use i because for some reason iterrows()
    # gives you items out of order
    priority = 1
    for i, box in ordered_boxes.iterrows():
        x,y = box['xmin'], box['ymin']
        w,h = box['xmax']-box['xmin'], box['ymax']-box['ymin']
        
        rendered_output = cv2.putText(img=rendered_output, 
                                      text=f'{priority}', 
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      org=(int(x+w/2-10), int(y+h/2+10)),
                                      fontScale=10,
                                      color=(0,255,0),
                                      thickness=15,
                                      lineType=cv2.LINE_AA)
        priority += 1

    cv2.imshow('Test Video', cv2.resize(rendered_output, (1100,750)))

    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()