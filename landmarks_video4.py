# coding:utf-8
import sys
import dlib
import numpy
from skimage import io
import cv2
import socket
from time import sleep

host = "192.168.11.26" #Processingで立ち上げたサーバのIPアドレス
port = 10001       #Processingで設定したポート番号

def main():
    ESC_KEY = 27     # Escキーのキーコード
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps
    color = (0, 0, 255)
    
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成
    socket_client.connect((host, port))                               #サーバに接続

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    
    cascade_face_pass = './haarcascades/haarcascade_frontalface_alt2.xml'
    cascade_face = cv2.CascadeClassifier(cascade_face_pass)
    
    cap = cv2.VideoCapture(0)
    end_flag, frame = cap.read()
        
    while end_flag == True:
        #反転
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
        #右こめかみ
        x3 = shape.part(15).x
        y3 = shape.part(15).y
        #左こめかみ
        x4 = shape.part(1).x
        y4 = shape.part(1).y
        #面積（絶対値）
        sRight = abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
        sLeft  = abs(0.5*(x1*(y2-y4)+x2*(y4-y1)+x4*(y1-y2)))
        #scoreは約-5.0~5.0
        score = round((sLeft - sRight)/150,1)
        if score > 1.5:
            print('右　left:{1}    right:{0}  score:{2}'.format(sLeft,sRight,score))
        elif score < -1.5:
            print('左　left:{1}    right:{0}   score:{2}'.format(sLeft,sRight,score))
        else:
            print('正面　left:{1}   right:{0}   score:{2}'.format(sLeft,sRight,score))

        socket_client.send(str(score).encode('utf-8')) #データをstr型として送信
        
        # Escキーで終了
        key = cv2.waitKey(10)
        if key == ESC_KEY:
            socket_client.send(str(0).encode('utf-8'))
            break

    
        #次のフレーム読み込み
        end_flag, frame = cap.read()

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()

