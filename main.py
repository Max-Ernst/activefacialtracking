import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker():
    def __init__(self, maxDisappeared = 50):
        

