import cv2
import matplotlib.pyplot as plt
import numpy as np

# 画像の読み込み
image = cv2.imread("img/image3.png", cv2.IMREAD_GRAYSCALE)

# 水平方向微分のカーネル
kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# 垂直方向微分のカーネル
kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# フィルタ２Dを適用
prewitt_x = cv2.filter2D(image, -1, kernel_x)
prewitt_y = cv2.filter2D(image, -1, kernel_y)

#エッジの強度を計算
prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)