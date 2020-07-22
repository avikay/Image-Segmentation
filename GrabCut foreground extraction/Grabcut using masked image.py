# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:13:36 2020

@author: Avinash
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

newmask = cv2.imread("masked1.jpg",0)

mask[newmask == 0] = 0
mask[newmask == 1] = 1

mask, bgdModel, fgdModel = cv2.grabCut(img,)