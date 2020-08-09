"""
@File:    data_loader
@Author:  GongJintao
@Create:  8/6/2020 1:02 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import re
import torch
import numpy as np
import pandas as pd
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class ImdbDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.max_text_len = config.max_text_len
        self.vocab_size = config.vocab_size
        self.data = pd.DataFrame()
        self.data_process()

    def __getitem__(self, item):
        return torch.LongTensor(self.data.loc[item, 'x']), torch.tensor(
            self.data.loc[item, 'y'])

    def __len__(self):
        return len(self.data)

    def text_clean(self, text):
        eng_stopwords = stopwords.words('english')  # 停用词
        text = BeautifulSoup(text, 'html.parser').get_text()  # 去除html标签
        text = re.sub(r'[^a-zA-Z]', ' ', text)  # 去除标点
        words = text.lower().split()  # 全部转成小写，然后按空格分词
        words = [w for w in words if w not in eng_stopwords]  # 去除停用词
        return ' '.join(words)  # 重组成新的句子

    def data_process(self):
        df = pd.read_csv(self.data_dir+'labeledTrainData.tsv', sep='\t', escapechar='\\')
        df['clean_review'] = df['review'].apply(self.text_clean)

        self.vocab = self.get_vocab(df['clean_review'].tolist())
        self.word2id = {w: i + 1 for i, w in enumerate(self.vocab)}
        self.word2id['<UNK>'] = 0

        self.data['x'] = df['clean_review'].apply(self.padding_length)
        self.data['y'] = df['sentiment']

    def get_vocab(self, sentences):
        word_list = " ".join(sentences).split()
        if self.vocab_size > len(set(word_list)):
            vocab = list(set(word_list))
        else:
            counter = Counter(word_list).most_common(self.vocab_size-1)
            vocab, _ = list(zip(*counter))
        return vocab

    def padding_length(self, text):
        text2token = [
            self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in text.split()]
        if len(text2token) >= self.max_text_len:
            text_id = text2token[:self.max_text_len]
        else:
            text_id = text2token + [self.word2id['<UNK>']] * (self.max_text_len - len(text2token))
        return text_id


class ImdbDataLoader(DataLoader):
    def __init__(
            self,
            dataset,
            config):

        self.shuffle = config.shuffle
        self.validation_split = config.validation_split

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': config.batch_size,
            'shuffle': self.shuffle,
            'num_workers': config.num_workers
        }
        super(
            ImdbDataLoader,
            self).__init__(
            sampler=self.sampler,
            **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):  # 整数形式
            assert split > 0
            assert split < self.n_samples
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)  # 分数形式

        valid_idx = idx_full[:len_valid]
        train_idx = idx_full[len_valid:]

        train_sampler = SubsetRandomSampler(
            train_idx)  # 根据传入的indices，从样本中无放回的随机抽样
        valid_sampler = SubsetRandomSampler(valid_idx)

        # 如果已经划分了数据集，且随机抽样了，就无需设置shuffle参数，且需要更新训练集长度
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


if __name__ == "__main__":
    CONFIG_PATH = "../config.json"
    from utils.utils import read_json, ConfigParser
    dict = read_json(CONFIG_PATH)
    config = ConfigParser(dict)

    dataset = ImdbDataset(config)
    train_loader = ImdbDataLoader(dataset, config)
    valid_loader = train_loader.split_validation()

    for index, item in enumerate(train_loader):
        print(item)
        print('-----------------------------------')
        if index == 2:
            break
