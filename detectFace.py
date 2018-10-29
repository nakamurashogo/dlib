# coding:utf-8
import cv2
from time import sleep


#動画での顔と目の検出プログラム
if __name__ == '__main__':
    ESC_KEY = 27     # Escキーのキーコード
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    # 分類器の指定
    cascade_face = "/Users/nakamurashogo/Downloads/opencv_data/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade_face = cv2.CascadeClassifier(cascade_face)

    # カメラ映像取得
    cap = cv2.VideoCapture(0)

    # 初期フレームの読込
    #end_flagにはTrueかFalseが入り、frameには動画（フレーム）が入る
    end_flag, frame = cap.read()

    #height, width, channels = c_frame.shape
    #「shape」でc_frameの配列を渡す。c_frame.shapeは(480,640,3)

    # ウィンドウの準備
    #cv2.namedWindow(org)
    #cv2.namedWindow(face)


    while end_flag == True:
        # 画像の取得と顔の検出
        img = frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_list = cascade_face.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

        
        for rect in face_list:
              img = img[rect[1]-25:rect[1]+rect[3]+40,rect[0]-25:rect[0]+rect[2]+25]
        

        print (face_list)
        cv2.imshow("org", img)
        
        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, frame = cap.read()

# 終了処理
cap.release()
cv2.destroyAllWindows()
