# -*- coding: utf-8 -*-
#

import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    net = LeNet5()
    print(net)