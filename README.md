# Python Sudoku Solver
Automatically solve a Sudoku after image extraction and digit recognition.

## Data
Using data from mnist dataset of Arab numbers and from self-made Chinese dataset.

By running the *Data_prepare.py* script, the pickle files will be generated under *data/pkl_cache*.

Data size (Chinese, Arab, Format): 
- 6722,	54077,	(60799, 28, 28)
- 1855,	9020,	(10875, 28, 28)

## Model
By running the *CNN_train_test.py* script, the model will be generated under *model*.

## Reference
Some of ideas for image processing comes from reference code, which is under *_ReferenceCode*.