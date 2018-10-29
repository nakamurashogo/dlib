# coding:utf-8
import sys
import dlib
import numpy
from skimage import io
import cv2

def main():
    ESC_KEY = 27     # Escキーのキーコード
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps
    color = (0, 0, 255)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    
    cascade_face = "/Users/nakamurashogo/Downloads/opencv_data/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade_face = cv2.CascadeClassifier(cascade_face)
    
    cap = cv2.VideoCapture(0)
    end_flag, frame = cap.read()
    time = 0
        
    while end_flag == True:
        time += 1
        #print(time)
        frame = cv2.flip(frame,1)
        img = frame
        
        #顔の検出
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade_face.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
        #切り取り式　img[y:x]
        if len(face_list) > 0:
            for rect in face_list:
                img = img[rect[1]-30:rect[1]+rect[3]+80,rect[0]-50:rect[0]+rect[2]+50]
        else:
            print("noface")
    
        #顔検出（四角）
        rects = detector(img, 1)
        #特徴点検出
        for rect in rects:
            landmarks = numpy.matrix([[p.x , p.y]
                                       for p in predictor(img,rect).parts()]
                                    )
        shape = predictor(img, rect)
        
        #描画
        for i in range(68):
           cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1,color, thickness=1)
        cv2.imshow("org", frame)

        #鼻の頂点
        x1 = shape.part(30).x
        y1 = shape.part(30).y
        #鼻の根元
        x2 = shape.part(27).x
        y2 = shape.part(27).y
        #右目
        x3 = shape.part(1).x
        y3 = shape.part(1).y
        #左目
        x4 = shape.part(15).x
        y4 = shape.part(15).y
        #面積（絶対値）
        sRight = abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
        sLeft  = abs(0.5*(x1*(y2-y4)+x2*(y4-y1)+x4*(y1-y2)))
        
        if sRight > sLeft:
            print('右')
        elif sLeft > sRight:
            print('左')
        else:
            print('正面')

        # Escキーで終了
        key = cv2.waitKey(10)
        if key == ESC_KEY:
            break
            
        #次のフレーム読み込み
        end_flag, frame = cap.read()

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()

