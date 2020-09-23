"""
@File:    train
@Author:  GongJintao
@Create:  8/7/2020 11:05 AM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.data_loader import ImdbDataset, ImdbDataLoader
from utils.utils import read_json, ConfigParser
from model.torch_textcnn import TextCnn


class Trainer:
    def __init__(self, config, model,data_loader, valid_data_loader=None):
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.epochs = config.epochs
        self.early_stop = config.early_stop
        self.do_validation = config.do_validation
        self.lr = config.lr

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_epoch(self, epoch):
        train_loss = 0.0
        results = {}
        for batch_idx, (data, target) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(self.data_loader.dataset),
                100. * batch_idx / len(self.data_loader), loss.item()))

        train_loss /= len(self.data_loader)
        results['train_loss'] = train_loss
        if self.do_validation:
            results.update(self._valid_epoch(epoch))

        return results

    def _valid_epoch(self, epoch):
        self.model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                valid_correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, valid_correct, len(self.valid_data_loader.dataset),
            100. * valid_correct / len(self.valid_data_loader.dataset)))
        return {
            'valid_loss': valid_loss / len(self.valid_data_loader),
            'valid_correct': 100. * valid_correct / len(self.valid_data_loader.dataset)
        }

    def train(self):
        not_improved_count = 0
        best_acc = 0
        for epoch in range(self.epochs):
            result = self._train_epoch(epoch)

            # TODO:1. 保存模型
            # TODO:2. 用TensorBoard 记录loss和acc

            if not isinstance(self.early_stop, int):
                continue

            if result['valid_correct'] < best_acc:
                best_acc = result['valid_correct']
                not_improved_count = 0
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                print(
                    "Validation performance didn\'t improve for {} epochs. "
                    "Training stops.".format(
                        self.early_stop))
                break


def main(config):
    dataset = ImdbDataset(config)
    train_loader = ImdbDataLoader(dataset, config)
    valid_loader = train_loader.split_validation()
    model = TextCnn(config)

    trainer = Trainer(config,model, train_loader, valid_loader)
    trainer.train()


if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    dict = read_json(CONFIG_PATH)
    config = ConfigParser(dict)

    main(config)
