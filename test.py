import cv2
from utils.img_preprocess import img_show


test_img_path = "./sudoku-test-image/test1/2-3.jpg"

img = cv2.imread(test_img_path, 0)

img_show(img)


kernel = None
erode = cv2.erode(img.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 10)

img_show(erode)
