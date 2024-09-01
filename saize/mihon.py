import numpy as np
import cv2
from matplotlib import pyplot as plt
from utlis.show_image import show_1Img, show_2Img
from utlis.clahe import apply_clahe_and_plot

# ここではグレースケールにした画像から特徴量抽出をしている
# そのほかのエッジ検出などはしていない
# 透視変換後に


MIN_MATCH_COUNT = 10

# 画像を読み込む
# 4が正常
# 3が不正常
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
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# claheで明るさを揃える
img1_clahe , img2_clahe = apply_clahe_and_plot(img1, img2)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_clahe, None)
kp2, des2 = sift.detectAndCompute(img2_clahe, None)

# 特徴量記述子のデータ型をfloat32に変換
des1 = des1.astype(np.float32)
des2 = des2.astype(np.float32)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

else:
    print('Not enough matches are found - {}/{}'.format(len(good_matches), MIN_MATCH_COUNT))
    exit()

h, w ,_ = image1.shape
image2_aligned = cv2.warpPerspective(image2, M, (w,h))
show_1Img(image2_aligned)

alpha = 0.5
blended_image = cv2.addWeighted(image1, alpha, image2_aligned, 1 - alpha, 0)



# 結果を表示する
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
