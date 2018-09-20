#! /usr/bin/python
# -*- coding: utf-8 -*-
u"""dlibによる顔画像検出."""
import cv2
import numpy
import dlib

# 画像ファイルパスを指定
img_path = 'face3.jpg'

detector = dlib.get_frontal_face_detector()
# RGB変換 (opencv形式からskimage形式に変換)
#"img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image = cv2.imread(img_path, cv2.IMREAD_COLOR)
# rectsの数だけ顔を検出
rects = detector(image, 1)
        
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

color = (0, 0, 255)

for rect in rects:
    landmarks = numpy.matrix(
        [[p.x , p.y]
         for p in predictor(image, rect).parts()]
    )
#print landmarks

shape = predictor(image, rect)
for i in range(68):
    cv2.circle(image, (shape.part(i).x, shape.part(i).y), 3, color, thickness=2)

cv2.imshow("image", image)

cv2.waitKey(10000)
cv2.destroyAllWindows()






