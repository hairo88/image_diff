from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# このコードはORBによる特徴点の検出、描画をしている
# 比較はしていない

parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
parser.add_argument('--input', help='Path to input image.', default='box.png')
args = parser.parse_args()

# 画像の読み込み
src = cv.imread(r'diff_3_1.png')
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# 特徴点の検出
# minHessian = 400
# detector = cv.xfeatures2d.SURF_create(hessianThreshold=minHessian)
# keypoints = detector.detect(src)

# ORBによる特徴点の検出
orb = cv.ORB_create()
keypoints = orb.detect(src, None)

# 特徴点の描画
img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
cv.drawKeypoints(src, keypoints, img_keypoints)

# 特徴点の表示
cv.imshow('SURF Keypoints', img_keypoints)
cv.waitKey()