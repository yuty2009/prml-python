# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.mnistreader import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
trainset = mnist.get_train_dataset(onehot_label=False,
                                   reshape=True, new_shape=(-1, imsize, imsize))
testset = mnist.get_test_dataset(onehot_label=False,
                                 reshape=True, new_shape=(-1, imsize, imsize))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :])  # r_out (batch, time_step, hidden_size)
        return out


model = RNN(input_size=imsize, hidden_size=64, num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train
epochs = 20
batch_size = 100
epoch_steps = np.ceil(trainset.num_examples / batch_size).astype('int')
for epoch in range(epochs):
    for step in range(epoch_steps):
        X_batch, y_batch = trainset.next_batch(batch_size)
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)

        yp_batch = model(X_batch)
        loss = F.cross_entropy(yp_batch, y_batch.long(), reduction='sum')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 50 steps
        if (step + 1) % 50 == 0:

            correct_test = 0.
            while testset.epochs_completed <= 0:
                X_batch, y_batch = testset.next_batch(1000)
                X_batch = torch.tensor(X_batch, device=device)
                y_batch = torch.tensor(y_batch, device=device)
                yp_batch = model(X_batch)
                # get the index of the max log-probability
                yp_batch = yp_batch.argmax(dim=1, keepdim=True).squeeze()
                correct_test += sum(yp_batch == y_batch)
            acc_test = correct_test / testset.num_examples
            testset.reset()

            print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Test Accuracy {:.4f}"
                  .format(epoch + 1, epochs,
                          (step + 1) * batch_size, trainset.num_examples,
                          loss.item(),
                          acc_test))

X_batch, y_batch = testset.next_batch(10)
X_batch = torch.tensor(X_batch, device=device)
y_batch = torch.tensor(y_batch, device=device)
yp_batch = model(X_batch)
yp_batch = yp_batch.argmax(dim=1, keepdim=True).squeeze()
print('Real number', y_batch.cpu().numpy())
print('Pred number', yp_batch.cpu().numpy())