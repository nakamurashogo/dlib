#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
from time import sleep
host = "192.168.11.26" #Processingで立ち上げたサーバのIPアドレス
port = 10001       #Processingで設定したポート番号

if __name__ == '__main__':
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成
    socket_client.connect((host, port))                               #サーバに接続
    
    num = 1
    data = 1
    while num > 0:
        if num < 50:
            num += 1
            sleep(0.3)
            socket_client.send(str(data).encode('utf-8')) #データをstr型として送信
            print num
        else:
            data = 0
            sleep(0.3)
            socket_client.send(str(data).encode('utf-8'))
            print num
