import cv2
import numpy as np

# これはだめ

# 画像を読み込む
image = cv2.imread('chess/move_1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# エッジを検出
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 輪郭を検出
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# チェス盤を探す
for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:  # 四角形を探す
        x, y, w, h = cv2.boundingRect(approx)
        chessboard = image[y:y+h, x:x+w]
        break

# 切り出したチェス盤を表示
cv2.imshow('Chessboard', chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()