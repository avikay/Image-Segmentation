# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:16:54 2020

@author: Avinash
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import io, color, measure

#Reading the image
img = cv2.imread("homes.jpg",0)

#Denoising if needed Bilateral blur or  median blur works best
#filtered_img = cv2.bilateralFilter(img, 9, 50, 50)
#Seems like we do not need to denoise our image for now

#Thresholdig our image
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#CLeaning
kernel = np.ones((5,5), np.uint8)

eroded = cv2.erode(thresh, kernel, iterations = 1)
dilated = cv2.dilate(eroded, kernel, iterations = 1)

#Converting the image to the binary image
mask = dilated == 255

#label mask
s = [[1,1,1],[1,1,1],[1,1,1]]

labeled_mask, num_labeled = ndimage.label(mask, structure = s)

img2 = color.label2rgb(labeled_mask, bg_label = 0)


cv2.imshow("TIA Original",img)
#cv2.imshow("TIA",filtered_img)
cv2.imshow("TIA labeled mask",img2)
#io.imshow(mask)

cv2.waitKey(0)
cv2.destroyAllWindows()