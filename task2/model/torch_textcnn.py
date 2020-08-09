"""
@File:    torch_textcnn
@Author:  GongJintao
@Create:  8/6/2020 7:59 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import torch
import torch.nn as nn




class TextCnn(nn.Module):
    def __init__(self,config):
        super(TextCnn, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config.embedding_size, out_channels=config.num_filters,kernel_size=h), #[batch_size, num_filters, max_text_len-h+1]
                nn.BatchNorm1d(num_features=config.num_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=config.max_text_len-h+1) #[batch_size, num_filters*1]
            )
            for h in config.kernel_size
        ])

        self.fc = nn.Linear(in_features=config.num_filters *len(config.kernel_size),out_features=config.num_classes)
        self.dropout = nn.Dropout(0.5)
        # 分类
        self.sm = nn.Softmax(dim=1)

    def forward(self,x):
        # [batch_size, max_text_len]
        embed_x = self.embedding(x) #[batch_size,max_text_len,embedding_size]

        embed_x = embed_x.permute(0,2,1)# [batch_size,embedding_size,max_text_len]

        out = [conv(embed_x) for conv in self.conv] # out[i]: [batch_size, num_filters*1】

        # 拼接不同尺寸的卷积核运算出来的结果
        out = torch.cat(out,dim=1) #[batch_size, num_filters * len(filter_size)]
        out = out.view(-1,out.shape[1])

        out = self.fc(out)

        out = self.dropout(out)

        out = self.sm(out)

        return out

if __name__ == "__main__":
    from utils.utils import read_json,ConfigParser
    CONFIG_PATH = "../config.json"
    dict = read_json(CONFIG_PATH)
    config = ConfigParser(dict)

    model = TextCnn(config)
    x = torch.rand(32,100).type(torch.LongTensor)
    out = model(x)
    print(out.shape)



