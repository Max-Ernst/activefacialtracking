import centroidtracker
import numpy as np
import cv2
import math

class objectTracker():

    def __init__(self):
        self.centroid_tracker = centroidtracker.CentroidTracker()
        (H, W) = (None, None) 
        self.object_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Need haar classifier for docks
        self.num = 0

    def get_num_detect(self, ret, frame): # returns number of objects being detected in a single frame
        frame_detect = frame.copy()

        object_rect = self.object_cascade.detectMultiScale(frame_detect, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))
        rects = []

        for (x,y,w,h) in object_rect:
            # create array for bounding boxes
            box = np.array([x,y,x+w,y+h], dtype = "int")
            rects.append(box)

            # update centroid tracker based off of bounding boxes
            objects = self.centroid_tracker.update(rects)

            self.num += 1

        return self.num
    
    def get_bearing(self, ret, frame): # returns 2D vector containing x dist and y dist to object relative to camera position in
        frame_detect = frame.copy()

        object_rect = self.object_cascade.detectMultiScale(frame_detect, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))
        rects = []

        for(x,y,w,h) in object_rect:
            box = np.array([x,y,x+w,y+h], dtype = "int")
            rects.append(box)

            objects = self.centroid_tracker.update(rects)

            for (objectID, centroid) in objects.items():
                #cv2.rectangle(frame_detect, (x, y), (x+w, y+h), (0, 0, 255), 5)
                x_from_center = centroid[0] - 320
                y_from_center = centroid[1] - 240
                dist = ((x_from_center**2)+(y_from_center**2))**(1/2)
                x_normalized = x_from_center/dist
                y_normalized = y_from_center/dist
                #cv2.circle(frame_detect, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                #cv2.arrowedLine(frame_detect, (320,240), (320+x_from_center, 240+y_from_center), (0, 0, 255), 1)
                #cv2.imshow('face', frame_detect)

                return (x_normalized, y_normalized)
                

    