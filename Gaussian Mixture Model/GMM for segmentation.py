# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:06:30 2020

@author: Avinash
"""

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

org_img = cv2.imread("Nepal_myland.jpg")
img_flat = org_img.reshape((-1,3)) 

gmm_model = GMM(n_components = 8, covariance_type = 'tied').fit(img_flat)
gmm_labels = gmm_model.predict(img_flat)

original_shape = (2160, 3840, 3)
segmented = gmm_labels.reshape(original_shape)

cv2.imwrite("Segmented image",segmented)
cv2.imshow("Segmented",segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
