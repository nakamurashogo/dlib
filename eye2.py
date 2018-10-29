# coding:utf-8
import cv2
import numpy as np


if __name__ == '__main__':
    
    cam=cv2.VideoCapture(0)

    while(cam.isOpened()):
        # カメラのフレームをキャプチャ
        ret, frame = cam.read()
        # ミラー
        frame = frame[:,::-1]
        # 入力画像をリサイズ
        #size = (640, 480)
        #frame = cv2.resize(frame, size)
        # グレースケール化
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 認識の精度を上げるために画像を平滑化
        gray = cv2.GaussianBlur(gray, (33,33), 1)
        
        # 表示用イメージ
        colimg = frame.copy() #cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 円検出
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1,minDist=2,
                                   param1=500,param2=30,minRadius=10,maxRadius=40)
        if circles != None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # 囲み線を描く
                cv2.circle(colimg,(i[0],i[1]),i[2],(255,255,255),3)
                # 中心点を描く
                cv2.circle(colimg,(i[0],i[1]),2,(0,0,255),3)
    
        # プレビュー
        cv2.imshow('preview', colimg)
        if cv2.waitKey(33) >= 0:
            break

    # ループを抜けたら終了
    cv2.destroyAllWindows()
