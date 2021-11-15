import centroidtracker
import numpy as np
import cv2

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
                x_from_center = centroid[0] - 320
                y_from_center = centroid[1] - 240

                return (x_from_center, y_from_center)
                

    