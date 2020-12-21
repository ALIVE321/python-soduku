import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm
from utils.img_preprocess import *
from tensorflow.keras.datasets import mnist
import ipykernel
from tensorflow.keras.preprocessing.image import img_to_array


chinese_numbers = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def roi(img):
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    return img


def handle_chinese_data():
    base_dir = "./data"
    for i in range(1, 11):
        folder = os.path.join(base_dir, "chinese", str(i))
        stand_dir = os.path.join(base_dir, "chinese_standard")
        if not os.path.exists(stand_dir):
            os.mkdir(stand_dir)
        stand_dir = os.path.join(stand_dir, str(i))
        if not os.path.exists(stand_dir):
            os.mkdir(stand_dir)
            os.mkdir(os.path.join(stand_dir, "test"))
            os.mkdir(os.path.join(stand_dir, "train"))
        for file in tqdm(os.listdir(os.path.join(folder, "testing"))):
            imag = img_standard(cv2.imread(os.path.join(folder, "testing", file), 0))
            cv2.imwrite(os.path.join(stand_dir, "test", file), imag)
        for file in tqdm(os.listdir(os.path.join(folder, "training"))):
            imag = img_standard(cv2.imread(os.path.join(folder, "training", file), 0))
            cv2.imwrite(os.path.join(stand_dir, "train", file), imag)
        # print("%d finished." % i)


def pkl_store_chinese():
    if not os.path.exists("./data/pkl_cache"):
        os.mkdir("./data/pkl_cache")
    base_dir = "./data/chinese_standard"
    train_data, train_label, test_data, test_label = [], [], [], []
    for i in range(1, 10):
        label = chinese_numbers[i]
        folder = os.path.join(base_dir, str(i))
        test_folder = os.path.join(folder, "test")
        for file in tqdm(os.listdir(test_folder)):
            test_data.append(roi(cv2.imread(os.path.join(test_folder, file), 0)))
            test_label.append(label)
        train_folder = os.path.join(folder, "train")
        for file in tqdm(os.listdir(train_folder)):
            train_data.append(roi(cv2.imread(os.path.join(train_folder, file), 0)))
            train_label.append(label)
    data = ((train_data, train_label), (test_data, test_label))
    with open("./data/pkl_cache/chinese_data.pkl", "wb") as fout:
        pickle.dump(data, fout)


def handle_mnist_data():
    ((train_data, train_label), (test_data, test_label)) = mnist.load_data()
    train_size = len(train_label)
    test_size = len(test_label)
    train_data = np.array([roi(train_data[i]) for i in range(train_size) if train_label[i] != 0])
    test_data = np.array([roi(test_data[i]) for i in range(test_size) if test_label[i] != 0])
    train_label = np.array([str(train_label[i]) for i in range(train_size) if train_label[i] != 0])
    test_label = np.array([str(test_label[i]) for i in range(test_size) if test_label[i] != 0])
    data = ((train_data, train_label), (test_data, test_label))
    with open("./data/pkl_cache/arab_data.pkl", "wb") as fout:
        pickle.dump(data, fout)


def merge_arab_chinese():
    with open("./data/pkl_cache/chinese_data.pkl", "rb") as fin:
        chinese_data = pickle.load(fin)
    with open("./data/pkl_cache/arab_data.pkl", "rb") as fin:
        arab_data = pickle.load(fin)
    merged_data = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            print(len(chinese_data[i][j]), len(arab_data[i][j]))
            merged_data[i][j] = np.concatenate((chinese_data[i][j], arab_data[i][j]))
            print(np.shape(merged_data[i][j]))
    merged_data[0] = tuple(merged_data[0])
    merged_data[1] = tuple(merged_data[1])
    merged_data = tuple(merged_data)
    with open("./data/pkl_cache/merged_data.pkl", "wb") as fout:
        pickle.dump(merged_data, fout)


if __name__ == "__main__":
    handle_chinese_data()
    pkl_store_chinese()
    handle_mnist_data()
    merge_arab_chinese()
