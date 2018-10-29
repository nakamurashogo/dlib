#! /usr/bin/python
# -*- coding: utf-8 -*-
import detectFace
import cv2


i = 1
while i > 0:
    
    frame = detectFace.detectFace()
    cv2.imshow("org", frame)
