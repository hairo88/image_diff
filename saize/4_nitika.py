import cv2
import numpy as np

from utlis.clahe import apply_clahe_and_plot

def onTrackbar(position):
    global threshold
    threshold = position

def nitika_image(img3):
    # ウィンドウを作成
    cv2.namedWindow("Thresholding")

    # トラックバーの初期設定
    global threshold  # グローバル変数を使用
    threshold = 100
    cv2.createTrackbar("track", "Thresholding", threshold, 255, onTrackbar)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_clahe = clahe.apply(img3)
    filter_image = cv2.bilateralFilter(image_clahe, 9, 75, 75)

    while True:

        # 通常の閾値による二値化
        ret, img_th_simple = cv2.threshold(img3, threshold, 255, cv2.THRESH_BINARY)

        # 適応的二値化（平均値）
        img_th_mean = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 適応的二値化（ガウシアン）
        img_th_gaussian = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 画像を真ん中で分ける
        height, width = img3.shape
        combined_image = np.zeros((height, width * 2), dtype=np.uint8)

        # 左半分に元の画像を表示
        combined_image[:, :width] = img3
        
        # 右半分に二値化結果を表示
        combined_image[:, width:] = img_th_simple  # ここは他の二値化結果に変更可能

        # ウィンドウに表示
        cv2.imshow("Thresholding", combined_image)

        # Escキーを押すとループ終了
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

img_3 = cv2.imread(r'img/image3.png', 0)  # グレースケールで読み込み
nitika_image(img_3)
