import torch
import torch.nn as nn
from model.crf import CRF


tags_set = [
    '<START>',
    '<STOP>',
    'I-PER',
    'O',
    'S-PER',
    'B-PER',
    'B-ORG',
    'B-MISC',
    'S-ORG',
    'E-MISC',
    'E-ORG',
    'B-LOC',
    'E-LOC',
    'I-ORG',
    'I-LOC',
    'S-MISC',
    'E-PER',
    'S-LOC',
    'I-MISC']


class BiLstmCrf(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            bidrection,
            dropout_rate,
            num_tags):
        super(BiLstmCrf, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidrection,
            num_layers=1,
            dropout=dropout_rate,
            batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x):
        #         [batch_size,seq_len,embedding_size]
        output, _ = self.lstm(x)  # [batch_size,seq_len,hidden_size*2]
        output = self.hidden2tag(output)
        output = self.crf.decode(output)
        return output


model = BiLstmCrf(130, 200, True, 0.5, 19)
input_x = torch.rand(10, 20, 130)
output = model(input_x)
