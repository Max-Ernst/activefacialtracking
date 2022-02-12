import numpy as np
import cv2
import os
import trackerClass

vid = cv2.VideoCapture(0) # this starts the camera, call this in initialization

object_tracker = trackerClass.objectTracker() # initializes object tracker, call this in initialization

# each time you wish to obtain a bearing, call the following two lines
ret, frame = vid.read() # this line takes a screenshot from the video
bearing = object_tracker.get_bearing(ret, frame) # get_bearing returns a 2D normalized vector giving the direction from the center of the screen to the object