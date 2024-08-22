import cv2
import numpy as np
import datetime
from sobel_def import sobel_image
from canny import canny_image
import matplotlib.pyplot as plt
from utlis.show_image import show_2Img, show_1Img
from utlis.clahe import apply_clahe_and_plot

# 画像を読み込む
image1 = cv2.imread(r'img/image4.png')
image2 = cv2.imread(r'img/image3.png')

# リサイズ
h_1, w_1, _= image1.shape
h_2, w_2, _= image2.shape

# 縦、横サイズ、それぞれ小さい方を見つける
h_min = min(h_1, h_2)
w_min = min(w_1, w_2)
image1 = image1[0:h_min, 0:w_min]
image2 = image2[0:h_min, 0:w_min]


# グレースケールに変換
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# claheで明るさを揃える
image1_clahe, image2_clahe = apply_clahe_and_plot(gray1, gray2)


# ノイズ除去
filtered_image1 = cv2.bilateralFilter(image1_clahe, 9, 75, 75)
filtered_image2 = cv2.bilateralFilter(image2_clahe, 9, 75, 75)


# 二値化
ret_1, image1_binary = cv2.threshold(filtered_image1, 73, 255, cv2.THRESH_BINARY)
ret_2, image2_binary = cv2.threshold(filtered_image2, 73, 255, cv2.THRESH_BINARY)

# show_2Img(image1_binary, image2_binary, 'binary')

 
# エッジ検出 
# 二値化した画像を使わずにエッジする
img_contour_only_1 = sobel_image(filtered_image1)
img_contour_only_2 = sobel_image(filtered_image2)

show_2Img(img_contour_only_1, img_contour_only_2, 'conttour')


#モルフォロジー演算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

kernel_2 = np.ones((5,5), np.uint8)
image1_mor = cv2.erode(img_contour_only_1, kernel_2, iterations =3)

image1_mor = cv2.morphologyEx(img_contour_only_1, cv2.MORPH_OPEN, kernel)
image2_mor = cv2.morphologyEx(img_contour_only_2, cv2.MORPH_OPEN, kernel)

show_2Img(image1_mor, image2_mor, 'mor')

# akaze検出器
akaze = cv2.AKAZE_create()
keypoints1, descriptors1 = akaze.detectAndCompute(image1_mor, None)
keypoints2, descriptors2 = akaze.detectAndCompute(image2_mor, None)

#透視変換(image2)

#BFMatcher型のオブジェクトを作成する
#ブルーとフォース抹茶ーを使用して距離感のマッチを探す
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 距離に基づいてマッチをソート（距離が小さいほど良い）
matches = sorted(matches, key=lambda x: x.distance)

#マッチしたもの中から上位を参考にする
good = matches[:int(len(matches) * 0.50)]

#対応が取れた特徴点の座標を取り出す
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# findhomegraphy:二つの画像からえられた点の集合からその物体の投射変換を計算する
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
#iamge2を透視変換する
image2_transform = cv2.warpPerspective(img_contour_only_2, M, (w_min, h_min))

#マッチを描画する
matched_image = cv2.drawMatches(img_contour_only_1, keypoints1, image2_transform, keypoints2, matches[:int(len(matches) * 0.50)], None, flags=2)

show_1Img(matched_image)

# show_2Img(img_contour_only_1, image2_transform, 'test')



# 画像を小さな領域に分割して比較するための処理
block_size = 100  # 小さな領域のサイズ

# 差分を格納する画像を作成する
difference = np.zeros_like(img_contour_only_1)

# 小領域ごとに差分を計算
for y in range(0, h_min, block_size):
    for x in range(0, w_min, block_size):
        # ブロックのサイズを調整
        block_h = min(block_size, h_min - y)
        block_w = min(block_size, w_min - x)

        block1 = img_contour_only_1[y:y+block_h, x:x+block_w]
        block2 = image2_transform[y:y+block_h, x:x+block_w]

        # 差分計算
        difference_block = cv2.absdiff(block1, block2)
        
        kernel = np.ones((3,3), np.uint8)
        difference_block_erode = cv2.erode(difference_block, kernel, iterations=1)
        difference_block_dilate = cv2.dilate(difference_block_erode, kernel, iterations=1)

        # 差分情報を差分画像に格納
        difference[y:y+block_h, x:x+block_w] = difference_block



# 違いを強調して表示する
cv2.imshow('Difference', difference)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 差分を強調するために閾値処理を行う
_, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# 差分画像を保存
# output_2 = f'output/akaze_diff_img{timestamp}.png'
# cv2.imwrite(output_2, thresh)


# 差分画像を表示
cv2.imshow('Thresholded Difference', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
