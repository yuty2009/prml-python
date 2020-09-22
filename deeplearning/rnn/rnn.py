# -*- coding: utf-8 -*-

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers= 1,
                           batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None) # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :]) # r_out (batch, time_step, hidden_size)
        return out

