# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:33:34 2020

@author: Avinash
"""

import cv2
import numpy as np

video = cv2.VideoCapture("highway.mp4")
subtractor = cv2.createBackgroundSubtractorMOG2(history = 20, varThreshold = 25, detectShadows = True)


while True:
    _, frame = video.read()
    
    mask = subtractor.apply(frame)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(30)
    if key == 27:
        break
    
video.release()
cv2.destroyAllWindows()