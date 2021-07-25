import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(
            self,
            chars_vocab_size,
            char_embed_size,
            kernel_size,
            num_filters,
            max_word_length,
            dropout_rate,
            batch_first=False):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.num_filters = num_filters

        self.batch_first = batch_first
        self.char_embedding = nn.Embedding(chars_vocab_size, char_embed_size)
        bias = np.sqrt(3.0 / char_embed_size)
        self.char_embedding.weight.data.uniform_(-bias, bias)  # 初始化嵌入层权重

        self.dropout = nn.Dropout(dropout_rate)

        self.conv = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=num_filters,
            kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=max_word_length - kernel_size + 1)

    def forward(self, x):
        # x: [max_seq_length,max_word_length]
        # [max_seq_length,max_word_length,char_embed_size]
        x = self.char_embedding(x)
        x = self.dropout(x)
        batch_size, seq_len,max_word_length,_ = x.shape
        view_shape = (batch_size * seq_len, max_word_length, self.char_embed_size)
        x = x.view(view_shape).transpose(1, 2)

        x = self.conv(x)  # [max_seq_length,num_filters]
        x = self.maxpool(x)
        output = x.view(batch_size,seq_len,self.num_filters)
        return output


if __name__ == "__main__":
    model = CNN(75, 30, 3, 30, 50, 0.5)
    input_x = torch.rand(10,20,50).type(torch.LongTensor)
    output = model(input_x)
    print(output.shape)
