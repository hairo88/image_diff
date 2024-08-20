import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


# imgA = cv2.imread('diff_3_1.png')
# imgB = cv2.imread('diff_3_2.png')

imgA = cv2.imread(r'img3.png')
imgB = cv2.imread(r'img4.png')



# 画像が正しく読み込まれたかをチェック
if imgA is None:
    # print(f'Could not open or find the image: {args.inputA}')
    print("error1")
    exit(0)
if imgB is None:
    # print(f'Could not open or find the image: {args.inputB}')
    print("error2")
    exit(0)

if imgA.shape[2] == 4:
    imgA = cv2.cvtColor(imgA, cv.COLOR_RGBA2RGB)
if imgB.shape[2] == 4:  # imgB が RGBA 形式の場合
    imgB = cv.cvtColor(imgB, cv.COLOR_RGBA2RGB)

imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

heightA, widthA, cA = imgA.shape[:3]
heightB, widthB, cB = imgB.shape[:3]

akaze = cv2.AKAZE_create()

kpA, desA = akaze.detectAndCompute(imgA, None)
kpB, desB = akaze.detectAndCompute(imgB, None)

# imageBを透視変換する
# 透視変換: 斜めから撮影した画像を真上から見た画像に変換する感じ
# BFMatcher型のオブジェクトを作成する
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 記述子をマッチさせる。※スキャン画像(B2)の特徴抽出はforループ前に実施済み。
matches = bf.match(desA,desB)
# マッチしたものを距離順に並べ替える。
matches = sorted(matches, key = lambda x:x.distance)
# マッチしたもの（ソート済み）の中から上位★%（参考：15%)をgoodとする。
good = matches[:int(len(matches) * 0.15)]
# 対応が取れた特徴点の座標を取り出す？
src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# findHomography:二つの画像から得られた点の集合を与えると、その物体の投射変換を計算する
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0) # dst_img作成の際だけ使う。warpperspectiveの使い方がわかってない。
# imgBを透視変換。
imgB_transform = cv2.warpPerspective(imgB, M, (widthA, heightA))

# imgAとdst_imgの差分を求めてresultとする。グレースケールに変換。
result = cv2.absdiff(imgA, imgB_transform)
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# 二値化
_, result_bin = cv2.threshold(result_gray, 50, 255, cv2.THRESH_BINARY) # 閾値は50

# カーネルを準備（オープニング用）
kernel = np.ones((2,2),np.uint8)
# オープニング（収縮→膨張）実行 ノイズ除去
result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel) # オープニング（収縮→膨張）。ノイズ除去。

# 二値画像をRGB形式に変換し、2枚の画像を重ねる。
result_bin_rgb = cv2.cvtColor(result_bin, cv2.COLOR_GRAY2RGB)
result_add = cv2.addWeighted(imgA, 0.3, result_bin_rgb, 0.7, 2.2) # ２.２はガンマ値。大きくすると白っぽくなる

cv2.imwrite('saize.jpg', result_add)
