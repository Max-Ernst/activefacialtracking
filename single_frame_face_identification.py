import numpy as np
import cv2
import os
import trackerClass

vid = cv2.VideoCapture(0)

object_tracker = trackerClass.objectTracker()

object_tracker.__init__()

ret, frame = vid.read()

bearing = object_tracker.get_bearing(ret, frame)

print(bearing)