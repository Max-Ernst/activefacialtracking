import centroidtracker
import numpy as np
import cv2

# initialize centroid tracker
centroid_tracker = centroidtracker.CentroidTracker()
(H, W) = (None, None)

# initialize cascade model
object_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# start video feed
vid = cv2.VideoCapture(0)

vid.set(3, 640)
vid.set(4, 480)

while(True):
    # get frame from video
    ret, frame = vid.read()
    frame_detect = frame.copy()

    # get x,y,w,h values from cascade model
    object_rect = object_cascade.detectMultiScale(frame_detect, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))

    rects = []

    for (x,y,w,h) in object_rect:
        # create array for bounding boxes
        box = np.array([x,y,x+w,y+h], dtype = "int")
        rects.append(box)

        # update centroid tracker based off of bounding boxes
        objects = centroid_tracker.update(rects)

        for (objectID, centroid) in objects.items():
            # label frame with info gained from cv
            text = "Object {}".format(objectID)
            cv2.putText(frame_detect, text, (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(frame_detect, (x, y), (x+w, y+h), (0, 0, 255), 5)
            cv2.circle(frame_detect, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

            x_from_center = centroid[0] - 320
            y_from_center = centroid[1] - 240

            cv2.putText(frame_detect, x_from_center, (x + int(w/2) - 40, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame_detect, y_from_center, (x + int(w/2) + 40, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)



    
    # output frame
    cv2.imshow('face', frame_detect)

    # press q to close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break