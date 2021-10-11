import centroidtracker
import numpy as np
import cv2
import matplotlib.pyplot as plt

ct = centroidtracker.CentroidTracker()
(H, W) = (None, None)

ct.__init__()

object_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def adjusted_face_detect(img):
    object_img = img.copy()
    object_rect = object_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)

    for (x,y,w,h) in object_rect:
        cv2.rectangle(object_img, (x, y), (x+w, y+h), (255, 0, 0), 5)

    return object_img

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    frame_detect = adjusted_face_detect(frame)
    
    cv2.imshow('frame', frame_detect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break