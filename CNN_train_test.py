from utils.CNN_Model import CnnNet
import numpy as np
import os
from utils.eval_metrics import *
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def draw_chart(loss, acc):
    assert len(loss) == len(acc)
    fig = plt.figure(figsize=(12, 8))
    plt.title("Loss/Acc -- Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # savgol_filter(y, 31, 4)
    x = [_ for _ in range(1, len(loss) + 1)]
    ax = fig.add_subplot(111)
    ax.plot(x, loss, '-', label='Loss')
    ax2 = ax.twinx()
    ax2.plot(x, acc, '-r', label='Accuracy')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Loss")
    ax2.set_ylabel(r"Accuracy")
    plt.savefig('./source/train_result.png')
    plt.show()


def cnn_model_train(classes: list, data: tuple, save_path:  str, epochs: int = 50):
    (train_data, train_labels), (test_data, test_labels) = data
    train_data = train_data.reshape((train_data.shape[0], 28, 28))
    # train_data = train_data.astype("float32") / 255.0

    test_data = test_data.reshape((test_data.shape[0], 28, 28))
    # test_data = test_data.astype("float32") / 255.0

    cnn = CnnNet(classes, 28)
    loss, acc = cnn.train(train_data, train_labels, test_data, test_labels, epochs, save_path)

    draw_chart(loss, acc)


def cnn_model_test(classes: list, data: tuple, load_path: str):
    (test_data, test_labels) = data[1]
    test_data = test_data.reshape((test_data.shape[0], 28, 28))
    # test_data = test_data.astype("float32") / 255.0

    cnn = CnnNet(classes, 28)
    acc, preds = cnn.test(test_data, test_labels, load_path=load_path)

    eval_metric(test_labels, preds, classes)


def load_data(load_path: str) -> tuple:
    with open(load_path, "rb") as fin:
        data = pickle.load(fin)
    return data


if __name__ == "__main__":
    feed_data = load_data("./data/pkl_cache/merged_data.pkl")

    class_list = [str(i) for i in range(1, 10)] + ["一", "二", "三", "四", "五", "六", "七", "八", "九"]

    # cnn_model_train(class_list, feed_data, "./models/model.save", 150)

    # feed_data = load_data("./data/pkl_cache/arab_data.pkl")
    # class_list = [str(i) for i in range(1, 10)]
    # cnn_model_train(class_list, feed_data, "./models/arab/model.save", 15)

    cnn_model_test(class_list, feed_data, "./models")

