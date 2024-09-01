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

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #画面上に描画する関数
    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    image2_transform = cv2.warpPerspective(img2_clahe, M, (w, h))
    
    #ノイズ
    image2_transform = cv2.GaussianBlur(image2_transform, (5,5), 0)
    show_1Img(image2_transform)


    # ---余白処理---
    #　ここに追加
    # image2_transform = cv2.addWeighted()


    # show_1Img(image2_transform)
    difference1 = cv2.absdiff(img1, img2)
    difference2 = cv2.absdiff(img1_clahe, image2_transform)
    show_2Img(difference1, difference2, 'difference')


else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)



img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()

#小領域ごとに差分を計算する
block_size = 100

difference = np.zeros_like(img1_clahe)

# 小領域ごとに差分を計算する
for y in range(0, h_min, block_size):
    for x in range(0, w_min, block_size):
        block_h = min(block_size, h_min - y)
        block_w = min(block_size, w_min - x)

        block1 = img1_clahe[y:y+block_h, x:x+block_w]
        block2 = image2_transform[y:y+block_h, x:x+block_w]

        # 差分計算
        difference_block = cv2.absdiff(block1, block2)
        
        kernel = np.ones((3,3), np.uint8)
        difference_block_erode = cv2.erode(difference_block, kernel, iterations=1)
        difference_block_dilate = cv2.dilate(difference_block_erode, kernel, iterations=1)

        # 差分情報を差分画像に格納
        difference[y:y+block_h, x:x+block_w] = difference_block_dilate
        # if y == 0:
        #     show_1Img(difference)

show_1Img(difference)

_, thresh = cv2.threshold(difference, 70, 255, cv2.THRESH_BINARY)

show_1Img(thresh)

#オープニング
# kernel = np.ones((5,5), np.uint8)
# img_opneing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# show_1Img(img_opneing)

# show_2Img(img1, image2_transform, 'diff')
# 差分を表示する

