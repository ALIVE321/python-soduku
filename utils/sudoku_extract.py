from utils.CNN_Model import CnnNet
import cv2
import numpy as np
from utils.img_preprocess import img_standard, img_show
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array


def sudoku_image(path: str, debug: bool = False) -> tuple:
    src_img = imutils.resize(cv2.imread(path), width=600)
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(grey_img, (7, 7), 3)
    bin_img = cv2.adaptiveThreshold(blur_img,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    11, 2)
    image = cv2.bitwise_not(cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    inv_img = cv2.bitwise_not(bin_img)
    if debug:
        # img_show(src_img, "Source Image:")
        # img_show(grey_img, "Greyscale Image:")
        # img_show(blur_img, "Blurred Image:")
        # img_show(bin_img, "Binary Image:")
        img_show(inv_img, "Inverted Image:")
        img_show(image, "Backup Image:")
    return src_img, inv_img, image, grey_img


def extract_sudoku(src: np.array, img: np.array, image: np.array, grey: np.array, debug: bool = False) -> tuple:
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            sudoku_contour = approx
            break
    if sudoku_contour is None:
        print("Cannot Find A Soduku Image Here.")
        return None, None
    sudoku_original = four_point_transform(src, sudoku_contour.reshape(4, 2))
    sudoku_standard = four_point_transform(grey, sudoku_contour.reshape(4, 2))
    sudoku_clear = four_point_transform(image, sudoku_contour.reshape(4, 2))

    if debug:
        display_image = src.copy()
        cv2.drawContours(display_image, [sudoku_contour], -1, (0, 255, 0), 2)
        img_show(display_image, "Contours Image:")
        img_show(sudoku_original, "Transformed Sudoku Image:")
        img_show(sudoku_standard, "Transformed Sudoku Binary Blurred Image:")
        img_show(sudoku_clear, "Transformed Sudoku Binary Clear Image:")
    return sudoku_original, sudoku_standard, sudoku_clear


def extract_cell(img: np.array, image: np.array, debug: bool = False) -> list:
    height, width = np.shape(img)
    cells = []
    for x in range(9):
        for y in range(9):
            cell_up = max(0, x * height // 9)
            cell_down = min(height, (x + 1) * height // 9)
            cell_left = max(0, y * width // 9)
            cell_right = min(width, (y + 1) * width // 9)
            cells.append(img[cell_up: cell_down, cell_left: cell_right])
    if debug:
        cells_image = np.hstack(tuple([np.vstack(tuple([cv2.resize(cells[i * 9 + j], (60, 60))
                                                        for i in range(9)]))
                                       for j in range(9)]))
        img_show(cells_image, "Cells Image:")
    return cells


def cnn_model_predict(cells: list, path: str) -> tuple:
    length = len(cells)
    flags = [0 if i is None else 1 for i in cells]
    # nums = [str(i) for i in range(1, 10)]
    nums = [str(i) for i in range(1, 10)] + ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    cnn = CnnNet(nums, 28)
    cell_with_num = [i for i in cells if i is not None]
    print("There are %d cells with number." % sum(flags), len(cell_with_num))
    prob_res, pred_res = cnn.predict(cell_with_num, path)
    print(pred_res)
    preds = [None] * length
    probs = [None] * length
    digits = [None] * length
    num = 0
    for i in range(length):
        if flags[i]:
            # cv2.imwrite("./tmp/cell-%d.jpg" % i, cell_with_num[num] * 255)
            preds[i] = pred_res[num]
            probs[i] = prob_res[num]
            digits[i] = (nums.index(preds[i]) % 9) + 1
            num += 1
    assert(num == len(pred_res))
    return probs, preds, digits


def cell_standard(cell: np.array, debug: bool = False):
    thresh_a = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh_a)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return None
    # img_show(np.hstack((cell, thresh_a, thresh)), "Cell For Standardizing")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, contours, -1, 255, -1)
    # cv2.drawContours(mask, [contour], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.018:
        return None
    cell_handled = cv2.resize(cv2.bitwise_and(thresh, thresh, mask=mask), (28, 28))
    cell_handled = img_to_array(cell_handled.astype("float") / 255.0).reshape((28, 28))
    if debug:
        img_show(cell_handled, "Cell after Resize")
    return cell_handled


def extract_digit(cells: list, model_path: str, debug: bool = False):
    standard_cells = []
    for i in range(len(cells)):
        standard_cells.append(cell_standard(cells[i], False))
    probs, preds, digits = cnn_model_predict(standard_cells, model_path)
    if debug:
        print("Predictions:", preds)
    return probs, preds, digits


if __name__ == "__main__":
    pass
