import cv2
import numpy as np

def adjust_brightness(images):
    # 参照画像（最初の画像）の平均輝度を計算
    reference_brightness = np.mean(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY))
    
    adjusted_images = []
    for img in images:
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 現在の画像の平均輝度を計算
        current_brightness = np.mean(gray)
        
        # 明るさを調整
        alpha = reference_brightness / current_brightness
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        adjusted_images.append(adjusted)
    
    return adjusted_images

# 画像の読み込み
image1 = cv2.imread(r'img/image3.png')
image2 = cv2.imread(r'img/image4.png')

# 明るさを調整
adjusted_images = adjust_brightness([image1, image2])

# 結果の表示
for i, img in enumerate(adjusted_images):
    cv2.imshow(f'Adjusted Image {i+1}', img)
    cv2.imwrite(f'AdjustedImage{i+1}.png', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()