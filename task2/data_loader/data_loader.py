"""
@File:    data_loader
@Author:  GongJintao
@Create:  8/6/2020 1:02 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class ImdbDataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data_process()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def text_clean(self,text):
        eng_stopwords = stopwords.words('english') #停用词
        text = BeautifulSoup(text, 'html.parser').get_text()  # 去除html标签
        text = re.sub(r'[^a-zA-Z]', ' ', text)  # 去除标点
        words = text.lower().split()  # 全部转成小写，然后按空格分词
        words = [w for w in words if w not in eng_stopwords]  # 去除停用词
        return ' '.join(words)  # 重组成新的句子

    def data_process(self):
        df = pd.read_csv(self.data_dir,sep='\t', escapechar='\\')
        df['clean_review'] = df['review'].apply(self.text_clean)
        self.data =  list(zip(df['clean_review'],df['sentiment']))

class ImdbDataLoader(DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
            validation_split=0.2,
            num_workers=1):

        self.shuffle = shuffle
        self.validation_split = validation_split

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler,self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size':batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super(ImdbDataLoader, self).__init__(sampler=self.sampler,**self.init_kwargs)

    def _split_sampler(self,split):
        if split == 0.0:
            return None,None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split,int): #整数形式
            assert split > 0
            assert split < self.n_samples
            len_valid = split
        else:
            len_valid = int(self.n_samples * split) #分数形式

        valid_idx = idx_full[:len_valid]
        train_idx = idx_full[len_valid:]

        train_sampler = SubsetRandomSampler(train_idx) #根据传入的indices，从样本中无放回的随机抽样
        valid_sampler = SubsetRandomSampler(valid_idx)

        # 如果已经划分了数据集，且随机抽样了，就无需设置shuffle参数，且需要更新训练集长度
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler,valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler,**self.init_kwargs)


if __name__ == "__main__":
    data_dir = "../data/IMDB/labeledTrainData.tsv"
    dataset = ImdbDataset(data_dir)
    train_loader = ImdbDataLoader(dataset,batch_size=8)
    valid_loader = train_loader.split_validation()

    for index,item in enumerate(train_loader):
        print(item)
        print('-----------------------------------')
        if index == 2:
            break



