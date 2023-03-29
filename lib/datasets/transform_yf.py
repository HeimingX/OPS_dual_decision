import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch
import logging
from .randaug_yf import Rand_Augment


class Randaug(Rand_Augment):
    def __init__(self,  Numbers=3, Magnitude=7, max_Magnitude=10, transforms=None, p=1.0):
        super(Randaug, self).__init__(Numbers=Numbers, Magnitude=Magnitude,
                                      max_Magnitude=max_Magnitude, transforms=transforms,
                                      p=p)


