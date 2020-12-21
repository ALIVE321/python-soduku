from utils.sudoku_extract import *
from utils.sudoku_solve import sudoku_solution


def show_predict_img(imag, probability, predict):
    h, w = imag.shape[:2]
    chinese = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    for i in range(81):
        if predict[i] is None:
            continue
        char = predict[i]
        score = int(np.round(probability[i] * 100))
        r = i // 9
        c = i % 9
        x = h * r // 9
        y = c * w // 9
        if char in chinese:
            char = "\"" + str(chinese.index(char) + 1) + "\""
        note = "/{}".format(score)
        cv2.putText(imag, char, (y, x + h // 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(imag, note, (y, x + h // 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("prediction", w, h)
    cv2.imshow("prediction", imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_solution_img(imag, predict):
    if predict is None:
        return
    h, w = imag.shape[:2]
    predict = predict.reshape([81])
    x_bia = h // 27
    y_bia = w // 27
    for i in range(81):
        char = str(predict[i])
        r = i // 9
        c = i % 9
        x = h * r // 9 + x_bia
        y = c * w // 9 + y_bia
        cv2.putText(imag, char, (y, x + h // 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.namedWindow("solution", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("solution", w, h)
    cv2.imshow("solution", imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_img_path = "./sudoku-test-image/sudoku_puzzle.jpg"
    # test_img_path = "./sudoku-test-image/test5.jpg"
    model_path = "./models/all"
    # model_path = "./models/arab"
    debug = True

    src_img, img, image, grey = sudoku_image(test_img_path, debug=False)
    original, standard, clear = extract_sudoku(src_img, img, image, grey, debug=False)
    cells = extract_cell(standard, clear, debug=False)
    probs, preds, digits = extract_digit(cells, model_path, debug=False)

    show_predict_img(original.copy(), probs, preds)

    solution = sudoku_solution(digits)

    show_solution_img(original.copy(), solution)
