"""
@File:    utils
@Author:  GongJintao
@Create:  8/4/2020 9:36 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
from sklearn import metrics

# softmax函数
def softmax(scores):
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores) / sum_exp
    return softmax


# 预测函数
def predict(w,x):
    scores = np.dot(x, w.T)
    probs = softmax(scores)

    return np.argmax(probs, axis=1).reshape(-1, 1)

def evaluate(y_true,y_pred):
    precision = metrics.precision_score(y_true,y_pred)
    recall = metrics.recall_score(y_true,y_pred)
    f1_score = metrics.f1_score(y_true,y_pred)

    return precision,recall,f1_score


