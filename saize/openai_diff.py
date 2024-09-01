import numpy as np
import cv2
from matplotlib import pyplot as plt
from utlis.show_image import show_1Img, show_2Img
from utlis.clahe import apply_clahe_and_plot
from ni import nitika_image

# 画像を読み込む
image1 = cv2.imread(r'img/image3.png')
image2 = cv2.imread(r'img/image4.png')

# リサイズ
h_1, w_1, _ = image1.shape
h_2, w_2, _ = image2.shape
h_min = min(h_1, h_2)
w_min = min(w_1, w_2)
image1 = image1[0:h_min, 0:w_min]
image2 = image2[0:h_min, 0:w_min]

# グレースケールに変換
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# CLAHEで明るさを揃える
gray1_clahe, gray2_clahe = apply_clahe_and_plot(gray1, gray2)

# AKAZEで特徴量検出
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1_clahe, None)
kp2, des2 = akaze.detectAndCompute(gray2_clahe, None)

# ディスクリプタをfloat32に変換
des1 = np.float32(des1)
des2 = np.float32(des2)

# FLANNベースのマッチング
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio testによる良いマッチの抽出
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = gray1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    image2_aligned = cv2.warpPerspective(gray2_clahe, M, (w, h))

    # ノイズ除去
    image2_aligned = cv2.GaussianBlur(image2_aligned, (5, 5), 0)

    # 差分計算
    block_size = 50
    difference = np.zeros_like(gray1)
    difference_without_morph = np.zeros_like(gray1)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_h = min(block_size, h - y)
            block_w = min(block_size, w - x)
            block1 = gray1_clahe[y:y + block_h, x:x + block_w]
            block2 = image2_aligned[y:y + block_h, x:x + block_w]

            # 差分
            difference_block = cv2.absdiff(block1, block2)
            difference_without_morph[y:y + block_h, x:x + block_w] = difference_block

            # モルフォロジー演算
            kernel = np.ones((5, 5), np.uint8)
            diff_block_erode = cv2.erode(difference_block, kernel, iterations=1)
            diff_block_dilate = cv2.dilate(diff_block_erode, kernel, iterations=1)

            difference[y:y + block_h, x:x + block_w] = diff_block_dilate

    # モルフォロジー処理をしない差分画像を保存
    cv2.imwrite('difference_without_morph.png', difference_without_morph)

    # モルフォロジー処理後の差分画像を保存
    cv2.imwrite('difference_with_morph.png', difference)

    # しきい値処理で差分の強調
    _, thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # モルフォロジー演算（オープニング処理）でノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    img_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 輪郭の検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 元画像に間違い部分を赤色で塗りつぶし
    image1_with_diff = image1.copy()
    cv2.drawContours(image1_with_diff, contours, -1, (0, 0, 255), -1) # -1を指定して塗りつぶす

    # 結果を表示
    plt.imshow(cv2.cvtColor(image1_with_diff, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 差分が強調された画像を保存
    cv2.imwrite('image1_with_diff.png', image1_with_diff)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None
