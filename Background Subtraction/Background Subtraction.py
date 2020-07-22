# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:21:40 2020

@author: Avinash
"""

import cv2
import numpy as np

video = cv2.VideoCapture("highway.mp4")

_, frame1 = video.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (5,5), 0)

while True:
    _, frame_vid = video.read()
    frame_vid_gray = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2GRAY)
    frame_vid_gray = cv2.GaussianBlur(frame_vid_gray,(5,5), 0)
    
    difference = cv2.absdiff(frame1_gray, frame_vid_gray)
    
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    
    
    cv2.imshow("difference", difference)
    
    key = cv2.waitKey(30)
    if key == 27:
        break
    
video.release()
cv2.destroyAllWindows()