import cv2
import numpy as np
from matplotlib import pyplot as plt

def onTrackbar(position):
    global threshold
    threshold = position

# def nitika_image(img3):
#     # ウィンドウを作成
#     cv2.namedWindow("Simple Threshold")

#     # トラックバーの初期設定
#     global threshold  # グローバル変数を使用
#     threshold = 100
#     cv2.createTrackbar("track", "Simple Threshold", threshold, 255, onTrackbar)

#     while True:
#         # 通常の閾値による二値化
#         ret, img_th_simple = cv2.threshold(img3, threshold, 255, cv2.THRESH_BINARY)

#         # 適応的二値化（平均値）
#         # img_th_mean = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#         # 適応的二値化（ガウシアン）
#         # img_th_gaussian = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#         # ウィンドウに表示
#         cv2.imshow("Simple Threshold", img_th_simple)
#         # cv2.imshow("mean threadshold", img_th_mean)
#         # cv2.imshow("gaussina", img_th_gaussian)

#         # Escキーを押すとループ終了
#         if cv2.waitKey(10) == 27:
#             break

#     cv2.destroyAllWindows()

import cv2
import numpy as np

# トラックバーのコールバック関数
def on_trackbar(val):
    global threshold
    threshold = val

def nitika_image(img1, img2):
    # ウィンドウを作成
    cv2.namedWindow("Comparison")

    # トラックバーの初期設定
    global threshold
    threshold = 100
    cv2.createTrackbar("Track", "Comparison", threshold, 255, on_trackbar)

    while True:
        # 通常の閾値による二値化を画像1と画像2に適用
        img1_th_simple = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)[1]
        img2_th_simple = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)[1]

        # 画像をリサイズ
        img1_resized = cv2.resize(img1_th_simple, (320, 240))  # サイズを320x240に変更
        img2_resized = cv2.resize(img2_th_simple, (320, 240))  # 同様にリサイズ

        # 画像を横に並べる
        combined_image = np.hstack((img1_resized, img2_resized))

        # 画像を表示
        cv2.imshow("Comparison", combined_image)

        # Escキーを押すとループ終了
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()
# 画像の読み込み
img1 = cv2.imread('img/image3.png', 0)
img2 = cv2.imread('img/image4.png', 0)

# リサイズ
h_1, w_1 = img1.shape
h_2, w_2 = img2.shape

# 縦、横サイズ、それぞれ小さい方を見つける
h_min = min(h_1, h_2)
w_min = min(w_1, w_2)
img1 = img1[0:h_min, 0:w_min]
img2 = img2[0:h_min, 0:w_min]

nitika_image(img1, img2)


# img_3 = cv2.imread(r'img/image3.png', 0)  # グレースケールで読み込み
# nitika_image(img_3)
