import cv2
import numpy as np

def local_feature_matching(img1, img2, grid_size=(2, 2)):
    # SIFT特徴量抽出器の初期化
    sift = cv2.SIFT_create()

    # 画像をグリッドに分割
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    cell_h1, cell_w1 = h1 // grid_size[0], w1 // grid_size[1]
    cell_h2, cell_w2 = h2 // grid_size[0], w2 // grid_size[1]

    all_matches = []
    all_kp1 = []
    all_kp2 = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 各グリッドセルの領域を定義
            roi1 = img1[i*cell_h1:(i+1)*cell_h1, j*cell_w1:(j+1)*cell_w1]
            roi2 = img2[i*cell_h2:(i+1)*cell_h2, j*cell_w2:(j+1)*cell_w2]

            # 各領域で特徴点と特徴量を抽出
            kp1, des1 = sift.detectAndCompute(roi1, None)
            kp2, des2 = sift.detectAndCompute(roi2, None)

            # 特徴量マッチング
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio testを適用
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # マッチング結果を全体のリストに追加
            all_matches.extend(good_matches)
            all_kp1.extend(kp1)
            all_kp2.extend(kp2)

    # マッチング結果の描画 (ループの外側で実行)
    img_matches = cv2.drawMatches(img1, all_kp1, img2, all_kp2, all_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return all_matches, img_matches

# 画像の読み込み
img1 = cv2.imread('img/image4.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img/image3.png', cv2.IMREAD_GRAYSCALE)

# 局所領域でのマッチング実行
matches, img_matches = local_feature_matching(img1, img2)

# 結果の表示
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()