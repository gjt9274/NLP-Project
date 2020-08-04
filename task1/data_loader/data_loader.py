"""
@File:    data_loader
@Author:  GongJintao
@Create:  8/4/2020 10:22 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


eng_stopwords = stopwords.words('english')  # 定义停用词


def data_process(file_path):
    df = pd.read_csv(file_path, sep='\t', escapechar='\\')
    print('Number of samples:{}'.format(len(df)))
    df['clean_review'] = df.review.apply(text_clean)

    vectorizer_feq = CountVectorizer(max_features=5000)  # 取词频为前5000的词
    data_freq = vectorizer_feq.fit_transform(df.clean_review).toarray()
    print("词频为特征的文本-单词矩阵维度:", data_freq.shape)

    # 2. 使用bigram，作为文本特征
    vectorizer_bigram = CountVectorizer(
        ngram_range=(
            2,
            2),
        max_features=1000,
        token_pattern=r'\b\w+\b',
        min_df=1)
    data_bigram = vectorizer_bigram.fit_transform(df.clean_review).toarray()
    print("bi-gram为特征的文本-单词矩阵维度：", data_bigram.shape)

    # 2. 使用tfidf, 作为文本特征
    vectorizer_tfidf = TfidfVectorizer(max_features=5000)
    data_tfidf = vectorizer_tfidf.fit_transform(df.clean_review).toarray()
    print("TF-IDF为特征的文本-单词矩阵维度：", data_tfidf.shape)

    return data_freq, data_bigram, data_tfidf, df['sentiment'].values


def text_clean(text):
    text = BeautifulSoup(text, 'html.parser').get_text()  # 去除html标签
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # 去除标点
    words = text.lower().split()  # 全部转成小写，然后按空格分词
    words = [w for w in words if w not in eng_stopwords]  # 去除停用词
    return ' '.join(words)  # 重组成新的句子


# 定义数据批量生成器
def batch_generator(data, batch_size, shuffle=True):
    x, y = data
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)  # 打乱顺序

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
