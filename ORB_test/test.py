import cv2
import numpy as np

# ORBによる特徴点の検出と描画をそれぞれ結び付けている

# 画像を読み込む
image1 = cv2.imread(r'diff_3_1.png')
image2 = cv2.imread(r'diff_3_2.png')

# グレースケールに変換
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# ORB検出器を使用してキーポイントとディスクリプタを検出
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# ブルートフォースマッチャーを使用してディスクリプタ間のマッチを探す
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 距離に基づいてマッチをソート（距離が小さいほど良い）
matches = sorted(matches, key=lambda x: x.distance)

# マッチを描画
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)

# マッチした画像を保存
# 画像を保存する場所を変更する

cv2.imwrite('matched_keypoints.jpg', matched_image)

# 2つの画像の差分を求める
difference = cv2.absdiff(gray1, gray2)

# 差分を強調するために閾値処理を行う
_, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# 差分画像を保存
cv2.imwrite('difference.jpg', thresh)

# 差分画像を表示
cv2.imshow('Difference', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()