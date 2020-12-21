import numpy as np


class Blank:
    def __init__(self, r, c, b):
        self.r = r
        self.c = c
        self.b = b


board = [[0 for _ in range(9)] for __ in range(9)]
rows = [[] for _ in range(9)]
cols = [[] for _ in range(9)]
blocks = [[] for _ in range(9)]
blanks = []


def sudoku_init(digits: list) -> bool:
    global board, rows, cols, blocks, blanks
    for i in range(81):
        row = i // 9
        col = i % 9
        block = (row // 3) * 3 + col // 3
        if digits[i] is not None:
            num = int(digits[i])
            board[row][col] = num
            rows[row].append(num)
            cols[col].append(num)
            blocks[block].append(num)
        else:
            board[row][col] = None
            blanks.append(Blank(row, col, block))
    return sudoku_check()  # print("There's something wrong with this sudoku.")


def dup(x) -> bool:
    return len(x) > len(set(x))


def sudoku_check() -> bool:
    global rows, cols, blocks
    for i in rows + cols + blocks:
        if dup(i):
            return False
    return True


def blank_range(blank: Blank) -> set:
    range_list = set([i for i in range(1, 10)])
    diff = set(rows[blank.r]).union(set(cols[blank.c])).union(set(blocks[blank.b]))
    return range_list.difference(diff)


def sudoku_solve() -> bool:
    global board, rows, cols, blocks, blanks
    if not len(blanks):
        print("Sudoku Solved")
        return True
    if not sudoku_check():
        return False
    blanks.sort(key=lambda x: len(rows[x.r]) + len(cols[x.c]) + len(blocks[x.b]), reverse=True)
    blank = blanks[0]
    range_list = blank_range(blank)
    for num in range_list:
        blanks.remove(blank)
        rows[blank.r].append(num)
        cols[blank.c].append(num)
        blocks[blank.b].append(num)
        board[blank.r][blank.c] = num
        if sudoku_solve():
            return True
        blanks.append(blank)
        rows[blank.r].remove(num)
        cols[blank.c].remove(num)
        blocks[blank.b].remove(num)
        board[blank.r][blank.c] = None
    return False


def sudoku_solution(digits: list):
    global board
    if not sudoku_init(digits):
        print("There's Something Wrong with the Initial Sudoku!!!")
    elif not sudoku_solve():
        print("There's Something Wrong when Solving the Sudoku!!!")
    else:
        print(np.array(board))
        return np.array(board)
    return None


if __name__ == "__main__":
    digit_list = [8, None, None, None, 1, None, None, None, 9,
                  None, 5, None, 8, None, 7, None, 1, None,
                  None, None, 4, None, 9, None, 7, None, None,
                  None, 6, None, 7, None, 1, None, 2, None,
                  5, None, 8, None, 6, None, 1, None, 7,
                  None, 1, None, 5, None, 2, None, 9, None,
                  None, None, 7, None, 4, None, 6, None, None,
                  None, 8, None, 3, None, 9, None, 4, None,
                  3, None, None, None, 5, None, None, None, 8]

    sudoku_solution(digit_list)
