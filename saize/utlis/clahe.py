import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_clahe_and_plot(image1, image2, clip_limit=4.0, tile_grid_size=(8, 8)):
    # CLAHEの作成
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # CLAHEを適用
    result1 = clahe.apply(image1)
    result2 = clahe.apply(image2)

    # ヒストグラムの作成
    # histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    # histogram2 = cv2.calcHist([result1], [0], None, [256], [0, 256])
    
    # # ヒストグラムの可視化
    # plt.rcParams["figure.figsize"] = [12, 7.5]
    # plt.subplots_adjust(left=0.01, right=0.95, bottom=0.10, top=0.95)

    # # 1枚目の画像とヒストグラム
    # plt.subplot(221)
    # plt.imshow(image1, cmap='gray')gazou
    # plt.axis("off")
    # plt.subplot(222)
    # plt.plot(histogram1)
    # plt.xlabel('Brightness')
    # plt.ylabel('Count')

    # # 2枚目の画像とヒストグラム
    # plt.subplot(223)
    # plt.imshow(result1, cmap='gray')
    # plt.axis("off")
    # plt.subplot(224)
    # plt.plot(histogram2)
    # plt.xlabel('Brightness')
    # plt.ylabel('Count')

    # plt.show()

    return result1, result2