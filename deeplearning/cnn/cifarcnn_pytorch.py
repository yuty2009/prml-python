# -*- coding: utf-8 -*-
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARCNN(nn.Module):

    def __init__(self, input_shape=None, num_classes=10):
        super(CIFARCNN, self).__init__()
        if input_shape is None:
            input_shape = (32, 32, 3)
        self.conv1 = nn.Conv2d(input_shape[-1], 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop2 = nn.Dropout(0.25)

        self.dense1 = nn.Linear(2304, 512)
        self.drop3 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, num_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.drop2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = F.relu(self.dense1(x))
        x = self.drop3(x)
        return F.log_softmax(self.dense2(x))
