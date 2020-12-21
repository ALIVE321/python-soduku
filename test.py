import cv2
from utils.CNN_Model import CnnNet
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from utils.CNN_Model import CnnNet
import numpy as np
import os
from utils.eval_metrics import *
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(load_path: str) -> tuple:
    with open(load_path, "rb") as fin:
        data = pickle.load(fin)
    return data


class_list = [str(i) for i in range(1, 10)] + ["一", "二", "三", "四", "五", "六", "七", "八", "九"]

# img = cv2.imread("./tmp/cell-80.jpg", 0)
# img = img.astype("float") / 255.0
# print(img.shape)
# img = img_to_array(img).reshape((28, 28))
# print(img.shape)
# cnn = CnnNet(class_list, 28)
#
# print(cnn.predict([img], "./output/all-1024"))

data = load_data("./data/pkl_cache/merged_data.pkl")
(train_data, train_labels), (test_data, test_labels) = data
train_data = train_data.reshape((train_data.shape[0], 28, 28))
train_data = train_data.astype("float32") / 255.0
test_data = test_data.reshape((test_data.shape[0], 28, 28))
test_data = test_data.astype("float32") / 255.0


