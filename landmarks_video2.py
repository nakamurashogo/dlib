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

        
    while end_flag == True:
            
        # 画像の取得と顔の検出
        img = frame
        
        rects = detector(img, 1)
        print (rects)

        for rect in rects:
            landmarks = numpy.matrix([[p.x , p.y]
                                       for p in predictor(img,rect).parts()]
                                    )
    
        shape = predictor(img, rect)
        print(shape)
        
        for i in range(68):
           cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1,color, thickness=1)

        cv2.imshow("org", img)
        #cv2.imshow("img_face", img_face)
        #cv2.moveWindow('gray',680, 150)
            
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

