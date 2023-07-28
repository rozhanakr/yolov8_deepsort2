import os 
import cv2 
import random 
from ultralytics import YOLO
from tracker import Tracker


cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
tracker = Tracker()

detection_threshold = 0.5

ret, frame = cap.read()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
while (ret):
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r 
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id) 
            if score > detection_threshold and class_id == 0:
                 detections.append([x1, y1, x2, y2, score])
            
            tracker.update(frame, detections)
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cv2.imshow('frame', frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
# Destroy all the windows
cv2.destroyAllWindows()