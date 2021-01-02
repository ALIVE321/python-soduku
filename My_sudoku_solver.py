from utils.sudoku_extract import *
from utils.sudoku_solve import sudoku_solution


def show_predict_img(imag, probability, predict):
    h, w = imag.shape[:2]
    chinese = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    x_bia = h // 27
    y_bia = w // 27
    for i in range(81):
        r = i // 9
        c = i % 9
        x = h * r // 9 + x_bia
        y = c * w // 9 + y_bia
        if predict[i]:
            char = predict[i]
            score = int(np.round(probability[i] * 100))
            if char in chinese:
                char = "\"" + str(chinese.index(char) + 1) + "\""
            note = "/{}".format(score)
            cv2.putText(imag, char, (y, x + h // 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(imag, note, (y, x + h // 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("prediction", w, h)
    cv2.imshow("prediction", imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_solution_img(imag, probability, predict, sol):
    h, w = imag.shape[:2]
    chinese = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    x_bia = h // 27
    y_bia = w // 27
    sol_f = 1
    if sol is None:
        sol_f = 0
    else:
        sol = sol.reshape([81])
    for i in range(81):
        r = i // 9
        c = i % 9
        x = h * r // 9 + x_bia
        y = c * w // 9 + y_bia
        if predict[i] is None:
            if sol_f:
                char = str(sol[i])
                cv2.putText(imag, char, (y, x + h // 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            char = str(predict[i])
            score = int(np.round(probability[i] * 100))
            if char in chinese:
                char = "\"" + str(chinese.index(char) + 1) + "\""
            note = "/{}".format(score)
            cv2.putText(imag, char, (y, x + h // 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(imag, note, (y, x + h // 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.namedWindow("solution", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("solution", w, h)
    cv2.imshow("solution", imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_img_path = "./sudoku-test-image/sudoku_puzzle.jpg"
    test_img_path = "./sudoku-test-image/test1/2-5.jpg"
    model_path = "./models"
    debug = True

    src_img, img, image, grey = sudoku_image(test_img_path, debug=False)
    original, standard, clear = extract_sudoku(src_img, img, image, grey, debug=False)

    # Thickness in test images
    standard = cv2.erode(standard.copy(), None, 10)

    cells = extract_cell(standard, debug)

    probs, preds, digits = extract_digit(cells, model_path, debug)

    show_predict_img(original.copy(), probs, preds)

    while True:
        correct = input("Correct The prediction: (r, c, n / 'q' quit)\n")
        if correct == "q":
            break
        r, c, n = [int(i) for i in correct.split()]
        digits[r * 9 + c] = n
        preds[r * 9 + c] = n
        probs[r * 9 + c] = 1

    solution = sudoku_solution(digits)
    show_solution_img(original, probs, preds, solution)
