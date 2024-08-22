import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_image(image):
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) 

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # 水平方向と垂直方向の勾配を組み合わせて合成勾配を計算
    # 画像とエッジ画像を表示
    plt.rcParams["figure.figsize"] = [12,7.5]                           # ウィンドウサイズを設定
    title = "cv2.Sobel: codevace.com"
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return sobel_combined