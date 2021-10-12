import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker():
    def __init__(self, maxDisappeared = 50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    # registers a new centroid, and increments object ID
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    # deregisters a centroid
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    # updates centroid data
    def update(self, rects):

        # if there are no objects being tracked, continually increment num of frames
        # an object is out of frame, and deregister those out of frame for too long
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # initialize a list of input centroids
        inputCentroids = np.zeros((len(rects), 2), dtype = "int")

        for(i, (startX, startY, endX, endY)) in enumerate(rects):
            # initalize the center of the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # register all new objects in input centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # establish lists of IDs and centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # calc distance array
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]

            usedRows = set()
            usedCols = set()

            for(row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects