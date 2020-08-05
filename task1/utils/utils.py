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