import numpy as np
import cv2

def canny_image(image):

    med_val = np.median(image)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    img_edge1 = cv2.Canny(image, threshold1 = min_val, threshold2 = max_val)

    # cv2.imshow('img', img_edge1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_edge1