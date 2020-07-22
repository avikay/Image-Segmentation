# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:00:16 2020

@author: Avinash
"""

import cv2
import numpy as np

org_img = cv2.imread('Nepal_myland.jpg')
img32 = org_img.reshape((-1,3))
img32 = np.float32(img32)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 8
attempts = 10

ret, label, center = cv2.kmeans(img32, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((org_img.shape))

cv2.imwrite("Segmented image 8cluster.jpg",res2)
cv2.imshow('Segmented image', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()