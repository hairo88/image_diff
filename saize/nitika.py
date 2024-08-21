import cv2
import numpy as np
from matplotlib import pyplot as plt
from sobel_def import sobel_image

img = cv2.imread('img/image3.png',0)

# sobel_img = sobel_image(img)

area = [3,15,31,63,127,255]

for i in range(6):
    plt.subplot(2,3,i+1)
    new_img =  cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,area[i],0)
    plt.imshow(new_img,'gray')
    plt.title("block size is {}".format(area[i]))
    plt.xticks([]),plt.yticks([])

plt.show()