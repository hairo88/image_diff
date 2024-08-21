import cv2
import datetime
import numpy as np
import os

# 画像の読み込み
imgA = cv2.imread(r'img/no_change_2.png')
imgB = cv2.imread(r'img/no_change_3.png')

# 画像が正しく読み込まれたかをチェック
if imgA is None:
    print("error1")
    exit(0)
if imgB is None:
    print("error2")
    exit(0)

# 画像がRGBA形式の場合はRGBに変換
if imgA.shape[2] == 4:
    imgA = cv2.cvtColor(imgA, cv2.COLOR_RGBA2RGB)
if imgB.shape[2] == 4:
    imgB = cv2.cvtColor(imgB, cv2.COLOR_RGBA2RGB)

# 画像をBGRからRGBに変換
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

# AKAZE特徴量検出器を作成
akaze = cv2.AKAZE_create()

# 特徴量を計算
kpA, desA = akaze.detectAndCompute(imgA, None)
kpB, desB = akaze.detectAndCompute(imgB, None)

# 特徴量のマッチングを行う
bf = cv2.BFMatcher()
matches = bf.knnMatch(desA, desB, k=2)

# マッチングの精度が高いもののみを抽出
ratio = 0.6  # raito -> ratio に修正

good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 特徴量を描画
img3 = cv2.drawMatchesKnn(imgA, kpA, imgB, kpB, good, None, flags=2)

# 画像を保存
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = f'output/points_chess_site_diff_{timestamp}.png'

cv2.imshow('img', img3)
cv2.imwrite(output, img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
