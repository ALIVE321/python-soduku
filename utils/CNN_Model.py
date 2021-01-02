import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle

tf.logging.set_verbosity(tf.logging.ERROR)


class CnnNet:
    def __init__(self, classes: list, image_size: int, batch_size: int = 100, lr: float = 0.001, conv_size: int = 3):
        assert len(classes) == len(list(set(classes)))
        self.class_list = classes
        self.class_size = len(self.class_list)
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.conv_size = conv_size
        self.model_path = "./output/model.save"
        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size], name="x-place")
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.class_size], name="y-place")
        self.drop_prob = tf.placeholder(dtype=tf.float32, name="drop-prob")
        print(self.class_size, "classes:", self.class_list)

    @staticmethod
    def half(x: int) -> int:
        return (x + 1) // 2

    def construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, self.image_size, self.image_size, 1])
        conv_matrix1 = tf.Variable(tf.random_normal(shape=[self.conv_size, self.conv_size, 1, 32], stddev=0.01),
                                   dtype=tf.float32)
        conv_bias1 = tf.Variable(tf.random_normal(shape=[32], stddev=0.01), dtype=tf.float32)
        conv_ret1 = tf.nn.conv2d(x_input, filter=conv_matrix1, strides=[1, 1, 1, 1], padding='SAME')
        conv_ret1 = tf.nn.relu(tf.nn.bias_add(conv_ret1, conv_bias1))
        conv_ret1 = tf.nn.max_pool(conv_ret1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv_ret1 = tf.nn.dropout(conv_ret1, rate=self.drop_prob)
        length = self.half(self.image_size)
        conv_matrix2 = tf.Variable(tf.random_normal(shape=[self.conv_size, self.conv_size, 32, 64], stddev=0.01),
                                   dtype=tf.float32)
        conv_bias2 = tf.Variable(tf.random_normal(shape=[64], stddev=0.01), dtype=tf.float32)
        conv_ret2 = tf.nn.conv2d(conv_ret1, filter=conv_matrix2, strides=[1, 1, 1, 1], padding='SAME')
        conv_ret2 = tf.nn.relu(tf.nn.bias_add(conv_ret2, conv_bias2))
        conv_ret2 = tf.nn.max_pool(conv_ret2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv_ret2 = tf.nn.dropout(conv_ret2, rate=self.drop_prob)
        length = self.half(length)
        conv_matrix3 = tf.Variable(tf.random_normal(shape=[self.conv_size, self.conv_size, 64, 128], stddev=0.01),
                                   dtype=tf.float32)
        conv_bias3 = tf.Variable(tf.random_normal(shape=[128], stddev=0.01), dtype=tf.float32)
        conv_ret3 = tf.nn.conv2d(conv_ret2, filter=conv_matrix3, strides=[1, 1, 1, 1], padding='SAME')
        conv_ret3 = tf.nn.relu(tf.nn.bias_add(conv_ret3, conv_bias3))
        conv_ret3 = tf.nn.max_pool(conv_ret3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv_ret3 = tf.nn.dropout(conv_ret3, rate=self.drop_prob)

        length = np.square(self.half(length)) * 128
        conv_ret = tf.reshape(conv_ret3, shape=[-1, length])

        fc_matrix1 = tf.Variable(tf.random_normal(shape=[length, 1024], stddev=0.01), dtype=tf.float32)
        fc_bias1 = tf.Variable(tf.random_normal(shape=[1024], stddev=0.01), dtype=tf.float32)
        fc_ret1 = tf.nn.relu(tf.add(tf.matmul(conv_ret, fc_matrix1), fc_bias1))
        fc_ret1 = tf.nn.dropout(fc_ret1, rate=self.drop_prob)
        fc_matrix2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fc_bias2 = tf.Variable(tf.random_normal(shape=[1024], stddev=0.01), dtype=tf.float32)
        fc_ret2 = tf.nn.relu(tf.add(tf.matmul(fc_ret1, fc_matrix2), fc_bias2))
        fc_ret2 = tf.nn.dropout(fc_ret2, rate=self.drop_prob)
        fc_matrix3 = tf.Variable(tf.random_normal(shape=[1024, self.class_size], stddev=0.01), dtype=tf.float32)
        fc_bias3 = tf.Variable(tf.random_normal(shape=[self.class_size], stddev=0.01), dtype=tf.float32)
        fc_ret3 = tf.add(tf.matmul(fc_ret2, fc_matrix3), fc_bias3, name="cnn_ret")

        return fc_ret3

    def train(self, x: list, y: list, tx: list, ty: list, epoch: int = 30, save_path: str = None):
        assert x[0].shape == (self.image_size, self.image_size)
        x = np.array(x)
        y = np.array([[0 if i != self.class_list.index(j) else 1 for i in range(self.class_size)] for j in y]) \
            .reshape([-1, self.class_size])
        tx = np.array(tx)
        ty = np.array([[0 if i != self.class_list.index(j) else 1 for i in range(self.class_size)] for j in ty]) \
            .reshape([-1, self.class_size])
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)
        x_train, x_test, y_train, y_test = x, tx, y, ty
        train_size, test_size = len(x_train), len(x_test)
        print("Scale of Train and Test: ", train_size, test_size)
        if save_path is None:
            save_path = self.model_path
        else:
            self.model_path = save_path

        cnn_ret = self.construct()
        predicts = tf.argmax(tf.nn.softmax(cnn_ret), axis=1)
        facts = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, facts), dtype=tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cnn_ret, labels=self.y_place))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_step = optimizer.minimize(cost)
        accs, losses = [], []
        print("Start Training.")
        # indices = [i for i in range(train_size)]
        with tf.Session() as s:
            init = tf.global_variables_initializer()
            s.run(init)
            saver = tf.train.Saver()
            for i in range(epoch):
                # shuffle(indices)
                ave_loss = 0
                for j in tqdm(range(0, train_size, self.batch_size)):
                    idx_train = np.random.choice(train_size, self.batch_size, False)
                    # idx_train = [k for k in range(j, min(train_size, j + self.batch_size))]
                    # idx_train = [indices[k] for k in range(j, min(train_size, j + self.batch_size))]
                    x_in = x_train[idx_train]
                    y_in = y_train[idx_train]
                    _, loss = s.run([train_step, cost], feed_dict={self.x_place: x_in,
                                                                   self.y_place: y_in,
                                                                   self.drop_prob: 0.25})
                # idx_test = np.random.choice(test_size, self.batch_size, False)
                # x_tmp = x_test[idx_test]
                # y_tmp = y_test[idx_test]
                ave_loss += loss
                acc = s.run(accuracy, feed_dict={self.x_place: x_test,  # x_tmp,
                                                 self.y_place: y_test,  # y_tmp,
                                                 self.drop_prob: 0})
                print("Epoch: {:>3d},    Accuracy: {:>1.4f},     Loss: {:>3.5f}".format(i + 1, acc, loss))
                accs.append(acc)
                losses.append(loss)
                if not i % 10:
                    saver.save(s, save_path, global_step=i)
                if i > 30 and acc >= 0.999:
                    saver.save(s, save_path, global_step=i)
                    break
        print("End Training. Model Saved At", self.model_path)
        saver.save(s, save_path, global_step=epoch-1)
        return losses, accs

    def test(self, x: list, y: list, load_path: str = None):
        if load_path is None:
            load_path = self.model_path
        model = tf.train.latest_checkpoint(load_path)
        print("Start Testing. Load Model From", model)
        assert x[0].shape == (self.image_size, self.image_size)
        x = np.array(x)
        y = np.array([[0 if i != self.class_list.index(j) else 1 for i in range(self.class_size)] for j in y]) \
            .reshape([-1, self.class_size])
        cnn_ret = self.construct()
        predicts = tf.argmax(tf.nn.softmax(cnn_ret), axis=1)
        facts = tf.argmax(self.y_place, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, facts), dtype=tf.float32))
        loader = tf.train.Saver()
        with tf.Session() as s:
            init = tf.global_variables_initializer()
            s.run(init)
            loader.restore(s, model)
            acc, preds = s.run([accuracy, predicts],
                               feed_dict={self.x_place: x,
                                          self.y_place: y,
                                          self.drop_prob: 0})
            print("End Testing. Acc={:<1.4f}".format(acc))
            preds = [self.class_list[i] for i in preds]
            return acc, preds

    def predict(self, x: list, load_path: str = None):
        if load_path is None:
            load_path = self.model_path
        model = tf.train.latest_checkpoint(load_path)
        print("Start Predicting. Load Model From", model)
        assert x[0].shape == (self.image_size, self.image_size)
        x = np.array(x)
        cnn_ret = self.construct()
        predicts = tf.argmax(tf.nn.softmax(cnn_ret), axis=1)
        probability = tf.reduce_max(tf.nn.softmax(cnn_ret), reduction_indices=[1])
        loader = tf.train.Saver()
        with tf.Session() as s:
            init = tf.global_variables_initializer()
            s.run(init)
            loader.restore(s, model)
            probs, preds = s.run([probability, predicts], feed_dict={self.x_place: x,
                                                                     self.drop_prob: 0})
            print("Finish Prediction.")
            preds = [self.class_list[i] for i in preds]
            return probs, preds


if __name__ == "__main__":
    pass
