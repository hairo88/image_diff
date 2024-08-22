import cv2
import numpy as np
from matplotlib import pyplot as plt

def onTrackbar(position):
    global threshold
    threshold = position

def nitika_image(img3):
    # ウィンドウを作成
    cv2.namedWindow("Simple Threshold")

    # トラックバーの初期設定
    global threshold  # グローバル変数を使用
    threshold = 100
    cv2.createTrackbar("track", "Simple Threshold", threshold, 255, onTrackbar)

    while True:
        # 通常の閾値による二値化
        ret, img_th_simple = cv2.threshold(img3, threshold, 255, cv2.THRESH_BINARY)

        # 適応的二値化（平均値）
        img_th_mean = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 適応的二値化（ガウシアン）
        img_th_gaussian = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # ウィンドウに表示
        cv2.imshow("Simple Threshold", img_th_simple)
        cv2.imshow("mean threadshold", img_th_mean)
        cv2.imshow("gaussina", img_th_gaussian)

        # Escキーを押すとループ終了
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

img_3 = cv2.imread(r'img/image3.png', 0)  # グレースケールで読み込み
nitika_image(img_3)
