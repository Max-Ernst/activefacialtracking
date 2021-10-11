import centroidtracker
import numpy as np
import cv2
import matplotlib.pyplot as plt

centroid_tracker = centroidtracker.CentroidTracker()
(H, W) = (None, None)

object_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    frame_detect = frame.copy()
    object_rect = object_cascade.detectMultiScale(frame_detect, scaleFactor = 1.3, minNeighbors = 7, minSize = (30,30))

    rects = []

    for (x,y,w,h) in object_rect:
        box = np.array([x,y,x+w,y+h], dtype = "int")
        rects.append(box)
        objects = centroid_tracker.update(rects)

        for (objectID, centroid) in objects.items():
            text = "Object {}".format(objectID)
            cv2.putText(frame_detect, text, (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(frame_detect, (x, y), (x+w, y+h), (0, 0, 255), 5)
            cv2.circle(frame_detect, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
    
    cv2.imshow('face', frame_detect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break