"""
    Preprocessor for Imagines.
    Include:
    - size 28*28
    - *greyscale
    - rename
"""
import cv2
import numpy as np
import os
from tqdm import tqdm
import ipykernel


def img_show(imag, name="img", key=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    h, w = np.shape(imag)[:2]
    h *= int(600 / w)
    cv2.resizeWindow(name, 600, h)
    cv2.imshow(name, imag)
    cv2.waitKey(key)
    cv2.destroyAllWindows()


def img_standard(imag, inverse=True):
    if inverse:
        imag = 255 - imag
    _, imag = cv2.threshold(imag, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imag = cv2.resize(imag, (28, 28), interpolation=3)
    return imag


if __name__ == "__main__":
    # base_dir = os.path.join(os.path.abspath(os.path.pardir), "data\\mydata")
    # print(base_dir)
    # for num in os.listdir(os.path.join(base_dir, "raw")):
    #     if not os.path.exists(os.path.join(base_dir, "standard", num)):
    #         os.mkdir(os.path.join(base_dir, "standard", num))
    #     cnt = 1
    #     pwd = os.path.join(base_dir, "raw", num)
    #     for img in tqdm(os.listdir(pwd)):
    #         img = os.path.join(pwd, img)
    #         x = cv2.imread(img, 0)
    #         x = cv2.resize(x, (28, 28), interpolation=3)
    #         cv2.imwrite(os.path.join(base_dir, "standard", num, "%s-518030910391-%d.png" % (num, cnt)), x)
    #         cnt += 1
    pass
