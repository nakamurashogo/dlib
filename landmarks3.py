#! /usr/bin/python
# -*- coding: utf-8 -*-
u"""dlibによる顔画像検出."""
import cv2
import numpy
import dlib

# 画像ファイルパスを指定
img_path = 'face3.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

detector = dlib.get_frontal_face_detector()

# RGB変換 (opencv形式からskimage形式に変換)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# rectsの数だけ顔を検出
rects = detector(img, 1)
        
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

color = (0, 0, 255)

for rect in rects:
    landmarks = numpy.matrix(
        [[p.x , p.y]
         for p in predictor(img, rect).parts()]
    )
#print landmarks

#ここから目検出
cascade_path_eye = "/Users/nakamurashogo/Downloads/opencv_data/haarcascades/haarcascade_eye.xml"
cascade_eye = cv2.CascadeClassifier(cascade_path_eye)
facerect_eye = cascade_eye.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

#画像に印（円）付け
shape = predictor(img, rect)
for i in range(68):
    cv2.circle(img, (shape.part(i).x, shape.part(i).y), 3,
            color, thickness=2)
for (x,y,w,h) in facerect_eye:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
