import torch
import cv2
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', force_reload=True)

cap = cv2.VideoCapture('far_west_test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame)
    cv2.imshow('Test Video', np.squeeze(results.render()))

    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()