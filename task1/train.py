"""
@File:    train
@Author:  GongJintao
@Create:  8/4/2020 10:10 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
from utils.utils import softmax, predict, evaluate
from data_loader.data_loader import data_process, batch_generator


N_CLASSES = 2


def split_data(x, y, val_split=0.2):
    n_samples = x.shape[0]

    indices = np.random.permutation(n_samples)
    split = int(n_samples * (1 - val_split))
    training_idx = indices[:split]
    valid_idx = indices[split:]

    train_x = x[training_idx]
    train_y = y[training_idx]

    valid_x = x[valid_idx]
    valid_y = y[valid_idx]

    return train_x, train_y, valid_x, valid_y


def valid(w, val_x, val_y, batch_size=64):
    val_loss = []
    val_gen = batch_generator((val_x, val_y), batch_size, shuffle=False)

    for batch_x, batch_y in val_gen:
        scores = np.dot(batch_x, w.T)
        prob = softmax(scores)

        y_one_hot = np.eye(N_CLASSES)[batch_y]
        # 损失函数
        loss = - (1.0 / len(batch_x)) * np.sum(y_one_hot * np.log(prob))
        val_loss.append(loss)

    return np.mean(val_loss)


def train(
        train_x,
        train_y,
        valid_x,
        valid_y,
        lr=0.1,
        batch_size=128,
        epochs=5000,
        early_stop=None):

    n_features = train_x.shape[1]
    w = np.random.rand(N_CLASSES, n_features)

    train_all_loss = []
    val_all_loss = []

    not_improved = 0
    best_val_loss = np.inf
    best_w = None

    for epoch in range(epochs):
        training_gen = batch_generator((train_x, train_y), batch_size)
        train_loss = []
        for batch_x, batch_y in training_gen:
            scores = np.dot(batch_x, w.T)
            prob = softmax(scores)

            y_one_hot = np.eye(N_CLASSES)[batch_y]
            # 损失函数
            loss = - (1.0 / len(batch_x)) * np.sum(y_one_hot * np.log(prob))
            train_loss.append(loss)

            # 梯度下降
            dw = -(1.0 / len(batch_x)) * np.dot((y_one_hot - prob).T, batch_x)
            w = w - lr * dw

        val_loss = valid(w, valid_x, valid_y, batch_size)

        val_precision, val_recall, val_f1_score = evaluate(
            valid_y, predict(w, valid_x))

        print(
            "Epoch = {0},the train loss = {1:.4f}, the val loss = {2:.4f}, precision={3:.4f}%, recall={4:.4f}%, f1_score={4:.4f}%".format(
                epoch,
                np.mean(train_loss),
                val_loss,
                val_precision * 100,
                val_recall * 100,
                val_f1_score * 100)
        )

        train_all_loss.append(np.mean(train_loss))
        val_all_loss.append(val_loss)

        if not isinstance(early_stop, int):
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w = w
            not_improved = 0
        else:
            not_improved += 1

        if not_improved > early_stop:
            print("Validation performance didn\'t improve for {} epochs. "
                  "Training stops.".format(early_stop))
            break

    return best_w, train_all_loss, val_all_loss


if __name__ == "__main__":
    train_file = "data/IMDB/labeledTrainData.tsv"
    data_freq, data_bigram, data_tfidf, label = data_process(train_file)
    train_x, trian_y, val_x, val_y = split_data(data_tfidf, label)

    w, train_all_loss, val_all_loss = train(
        train_x, trian_y, val_x, val_y, early_stop=10)
