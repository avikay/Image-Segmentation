# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:22:41 2020

@author: Avinash
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure
from skimage.segmentation import clear_border

#Reading the image
img = cv2.imread("homes.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Thresholdig our image
ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#CLeaning
kernel = np.ones((5,5), np.uint8)
opening =cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)

#Clear border
opening = clear_border(opening)

#background
sure_bg = cv2.dilate(opening, kernel, iterations = 1)
#foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

#define markers for watershed
ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers + 10

markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

#-1 represents our boundaries which's been extracted using watershed
#use label2rgb to fill in colors within boundaries
img[markers == -1] = [0,255,255]
img2 = color.label2rgb(markers, bg_label = 0)

plt.imshow(markers, cmap = 'jet')


cv2.imshow("Overlay original image",img)
cv2.imshow("colored houses", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
