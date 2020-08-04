"""
@File:    utils
@Author:  GongJintao
@Create:  8/4/2020 9:36 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np

# 定义数据批量生成器


def batch_generator(data, batch_size, shuffle=True):
    X, Y = data
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)  # 打乱顺序

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]


# softmax函数
def softmax(scores):
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores) / sum_exp
    return softmax


# 预测函数
def predict(w, b, x):
    scores = np.dot(x, w.T) + b
    probs = softmax(scores)

    return np.argmax(probs, axis=1).reshape(-1, 1)


# 评估函数，包括精确率，召回率和F1值
def evaluate(w, val_x, val_y):
    val_loss = []
    val_gen = batch_generator((val_x, val_y), batch_size=32, shuffle=False)
    for batch_x, batch_y in val_gen:
        scores = np.dot(batch_x, w.T)
        prob = softmax(scores)

        y_one_hot = one_hot(batch_y)
        # 损失函数
        loss = - (1.0 / len(batch_x)) * np.sum(y_one_hot * np.log(prob))
        val_loss.append(loss)

    return np.mean(val_loss)


def one_hot(batch_y, n_classes=2):
    n = batch_y.shape[0]
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), batch_y.T] = 1
    return one_hot
