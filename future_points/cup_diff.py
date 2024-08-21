import cv2
import os

def imwrite(fileName, img, params=None):
    try:
        ext = os.path.splitext(fileName)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(fileName, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False

    except Exception as e:
        print(e)
        return False

# 画像の読み込み
image1 = cv2.imread('image3.png')
# image2 = cv2.imread('cup_nothing.png')
image2= cv2.imread('image4.png')

if image1 is None:
    print('error1')
    exit()
if image2 is None:
    print('error2')
    exit()

# グレースケールに変換
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#リサイズ
gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

# 差分計算
diff = cv2.absdiff(gray1, gray2)

# 差分画像を二値化
_, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 差分の割合を計算
non_zero_count = cv2.countNonZero(diff_thresh)
total_pixels = diff_thresh.size
diff_ratio = non_zero_count / total_pixels

dirname = os.path.dirname(__file__)

imwrite(dirname + '/output/diff_thresh.png', diff_thresh)

# 物体の有無を判断 (例: 1% 以上の差分があれば物体が消えたと判断)
if diff_ratio > 0.01:
    print("物体が消えた可能性があります。")
else:
    print("物体は消えていないか、検出できませんでした。")
