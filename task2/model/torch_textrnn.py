import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        self.rnn = nn.RNN(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True)
        self.fc = self.fc = nn.Linear(
            in_features=config.hidden_size * 2,
            out_features=config.num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size,max_text_len,embedding_size]
        out, _ = self.rnn(out)  # [batch_size, max_text_len, hidden_size*2]
        out = self.fc(out[:,-1,:])  # [batch_size,max_text_len,num_classes]
        out = self.softmax(out)  # [batch_size, num_classess]

        return out


class Config(object):
    def __init__(self):
        self.embedding_size = 256
        self.hidden_size = 512
        self.vocab_size = 5000
        self.dropout_rate = 0.2
        self.num_layers = 2
        self.num_classes = 2


if __name__ == "__main__":
    config = Config()
    model = TextRNN(config)
    x = torch.rand(32, 100).type(torch.LongTensor)
    out = model(x)
    print(out.shape)
