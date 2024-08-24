import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_contours(sLeftPictureFile, sRightPictureFile):
    # 画像を読み込む
    Lsrc = cv2.imread(sLeftPictureFile)
    Rsrc = cv2.imread(sRightPictureFile)

    # 明るさ調整

    # AKAZE特徴量検出器を作成
    akaze = cv2.AKAZE_create()

    # 特徴量の検出と特徴量ベクトルの計算
    keypoints_left, descriptors_left = akaze.detectAndCompute(Lsrc, None)
    keypoints_right, descriptors_right = akaze.detectAndCompute(Rsrc, None)

    # 総当たりマッチング
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors_left, descriptors_right)

    # マッチングされた特徴点の座標を格納する配列
    pts_src = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # ホモグラフィ行列を計算
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

    # 射影変換
    WarpedSrcMat = cv2.warpPerspective(Lsrc, H, (Rsrc.shape[1], Rsrc.shape[0]))

    # マッチング結果の描画
    img_matches = cv2.drawMatches(Lsrc, keypoints_left, Rsrc, keypoints_right, matches, None)
    cv2.imwrite("result/matches.jpg", img_matches)


    # 差分画像の生成と処理
    Lmat_planes = cv2.split(WarpedSrcMat)
    Rmat_planes = cv2.split(Rsrc)

    diff0 = cv2.absdiff(Lmat_planes[0], Rmat_planes[0])
    diff1 = cv2.absdiff(Lmat_planes[1], Rmat_planes[1])
    diff2 = cv2.absdiff(Lmat_planes[2], Rmat_planes[2])

    diff0 = cv2.medianBlur(diff0, 5)
    diff1 = cv2.medianBlur(diff1, 5)
    diff2 = cv2.medianBlur(diff2, 5)

    wise_mat = cv2.bitwise_or(diff0, diff1)
    wise_mat = cv2.bitwise_or(wise_mat, diff2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening_mat = cv2.morphologyEx(wise_mat, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    dilation_mat = cv2.dilate(opening_mat, kernel, iterations=1)

    _, dilation_mat = cv2.threshold(dilation_mat, 90, 255, cv2.THRESH_BINARY)

    # 結果の合成と表示
    dilation_color_mat = cv2.cvtColor(dilation_mat, cv2.COLOR_GRAY2BGR)

    LaddMat = cv2.addWeighted(WarpedSrcMat, 0.3, dilation_color_mat, 0.7, 0)
    RaddMat = cv2.addWeighted(Rsrc, 0.3, dilation_color_mat, 0.7, 0)

    # 画像を表示
    plt.subplot(121), plt.imshow(cv2.cvtColor(LaddMat, cv2.COLOR_BGR2RGB)), plt.title('Left Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(RaddMat, cv2.COLOR_BGR2RGB)), plt.title('Right Image')
    plt.show()

    # 画像を保存 (必要に応じて)
    cv2.imwrite("result/LaddMat.jpg", LaddMat)
    cv2.imwrite("result/RaddMat.jpg", RaddMat)

# 画像ファイルパスを指定
sLeftPictureFile = "img/image4.png"
sRightPictureFile = "img/image3.png"


# 関数を実行
find_contours(sLeftPictureFile, sRightPictureFile)

# 画像を読み込む
# image1 = cv2.imread(r'img/image4.png')
# image2 = cv2.imread(r'img/image3.png')

# # リサイズ
# h_1, w_1, _= image1.shape
# h_2, w_2, _= image2.shape

# # 縦、横サイズ、それぞれ小さい方を見つける
# h_min = min(h_1, h_2)
# w_min = min(w_1, w_2)
# image1 = image1[0:h_min, 0:w_min]
# image2 = image2[0:h_min, 0:w_min]

# find_contours(image1, image2)







