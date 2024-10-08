import numpy as np
import cv2
import os

# お手本のutlisである

def imread(fileName, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(fileName, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


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