import numpy as np
import cv2
from matplotlib import pyplot as plt
from utlis.show_image import show_1Img, show_2Img
from utlis.clahe import apply_clahe_and_plot
from ni import nitika_image

def tirm_image(img, trim_per=10):
    hei, wid = img.shape[:2]
    trim_hei = int(hei * trim_per / 100)
    trim_wid = int(wid * trim_per / 100)
    trimmed_img = img[trim_hei: hei - trim_hei, trim_wid:wid - trim_wid]
    return trimmed_img

# --- 設定 ---
image_path1 = r'img/image4.png'
image_path2 = r'img/image3.png'
block_size = 100
MIN_MATCH_COUNT = 5

# --- 関数：特徴量マッチングと差分画像生成 ---
def detect_and_diff(image1, image2, akaze):
    h, w, _ = image1.shape
    difference = np.zeros_like(image1)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_h = min(block_size, h - y)
            block_w = min(block_size, w - x)
            block1 = image1[y:y + block_h, x:x + block_w]
            block2 = image2[y:y + block_h, x:x + block_w]

            kp1, des1 = akaze.detectAndCompute(block1, None)
            kp2, des2 = akaze.detectAndCompute(block2, None)

            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = matches[:10]

                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    image2_aligned = cv2.warpPerspective(block2, M, (block_w, block_h))
                    diff_block = cv2.absdiff(block1, image2_aligned)
                    diff_block = cv2.cvtColor(diff_block, cv2.COLOR_BGR2GRAY)
                    ret, diff_block = cv2.threshold(diff_block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel = np.ones((5, 5), np.uint8)
                    diff_block = cv2.erode(diff_block, kernel, iterations=1)
                    diff_block = cv2.dilate(diff_block, kernel, iterations=1)
                    diff_block = cv2.cvtColor(diff_block, cv2.COLOR_GRAY2BGR)
                    difference[y:y + block_h, x:x + block_w] = diff_block
                else :
                    print('good no')
    return difference

# --- メイン処理 ---

# 画像を読み込む
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 元の AKAZE パラメータ
akaze_default = cv2.AKAZE_create()
difference_default = detect_and_diff(image1.copy(), image2.copy(), akaze_default)
cv2.imwrite("difference_default.png", difference_default)

# 調整後の AKAZE パラメータ
akaze_tuned = cv2.AKAZE_create(threshold=0.0001, nOctaves=5, nOctaveLayers=4)
difference_tuned = detect_and_diff(image1.copy(), image2.copy(), akaze_tuned)
cv2.imwrite("difference_tuned.png", difference_tuned)

# 結果表示 (コメントアウトを外して使用)
show_1Img(difference_default, "Default AKAZE")
show_1Img(difference_tuned, "Tuned AKAZE")