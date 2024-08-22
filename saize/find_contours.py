import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from skimage import io

img = io.imread('img/image3.png')

#解説3
# グレースケールに変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二値化（閾値を150に設定）
ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

#解説4
# 輪郭を検出
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#解説5
# 全て白の画像を作成
img_blank = np.ones_like(img) * 255
# 輪郭だけを描画（黒色で描画）
img_contour_only = cv2.drawContours(img_blank, contours, -1, (0,0,0), 3)
# 描画
plt.imshow(cv2.cvtColor(img_contour_only, cv2.COLOR_BGR2RGB))
plt.show()