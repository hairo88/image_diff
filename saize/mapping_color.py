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

# グレースケールに変換 (全体マッチング用)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# CLAHEで明るさを揃える
gray1_clahe, gray2_clahe = apply_clahe_and_plot(gray1, gray2)

# AKAZEで特徴量検出 (全体マッチング用)
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

    h, w, _ = image1.shape  # カラー画像のサイズを取得
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    image2_aligned = cv2.warpPerspective(image2, M, (w, h))  # カラー画像を変換

    # 差分計算
    block_size = 300
    difference = np.zeros_like(gray1)  # gray1と同じサイズの差分画像
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_h = min(block_size, h - y)
            block_w = min(block_size, w - x)
            block1 = image1[y:y + block_h, x:x + block_w]  # カラー画像のブロック
            block2 = image2_aligned[y:y + block_h, x:x + block_w]  # カラー画像のブロック

            # --- カラー画像で特徴量マッチング ---
            # AKAZEで特徴量検出 (カラー画像用)
            kp1_block, des1_block = akaze.detectAndCompute(block1, None)
            kp2_block, des2_block = akaze.detectAndCompute(block2, None)

            # マッチング (ここではBrute-Force Matcherを使用)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1_block is not None and des2_block is not None:
                matches_block = bf.match(des1_block, des2_block)

                # マッチング結果からimage2_alignedに対応する部分を切り出す
                if matches_block:
                    # マッチング点の座標を取得
                    src_pts_block = np.float32([kp1_block[m.queryIdx].pt for m in matches_block]).reshape(-1, 1, 2)
                    dst_pts_block = np.float32([kp2_block[m.trainIdx].pt for m in matches_block]).reshape(-1, 1, 2)

                    # 変換行列を計算
                    M_block, _ = cv2.findHomography(dst_pts_block, src_pts_block, cv2.RANSAC, 5.0)

                    # image2_alignedをimage1に対応する部分に変換
                    block2_warped = cv2.warpPerspective(block2, M_block, (block_w, block_h))

                    # block1と変換後の画像の差分を計算 (カラー画像の差分)
                    block_difference = cv2.absdiff(block1, block2_warped)

                    # カラー画像の差分をグレースケールに変換
                    block_difference_gray = cv2.cvtColor(block_difference, cv2.COLOR_BGR2GRAY)

                    # 差分をdifferenceに保存
                    difference[y:y + block_h, x:x + block_w] = block_difference_gray

            # --- 差分計算ここまで ---

    # モルフォロジー処理後の差分画像を保存
    cv2.imwrite('difference.png', difference)

    # しきい値処理で差分の強調
    _, thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # モルフォロジー演算（オープニング処理）でノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    img_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 輪郭の検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 元画像に間違い部分を赤色で塗りつぶし
    image1_with_diff = image1.copy()
    cv2.drawContours(image1_with_diff, contours, -1, (0, 0, 255), -1)

    # 結果を表示
    plt.imshow(cv2.cvtColor(image1_with_diff, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 差分が強調された画像を保存
    cv2.imwrite('image1_with_diff.png', image1_with_diff)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None