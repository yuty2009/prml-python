# -*- coding: utf-8 -*-

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from dataloader.mnistreader import *
from deeplearning.cnn.pytorch.lenet5 import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
# trainset = mnist.get_train_dataset()
# testset = mnist.get_test_dataset()
trainset = mnist.get_train_dataset(onehot_label=False,
                                   reshape=True, new_shape=(-1, imsize, imsize, 1),
                                   transpose=True, new_pos=(0, 3, 1, 2))
testset = mnist.get_test_dataset(onehot_label=False,
                                 reshape=True, new_shape=(-1, imsize, imsize, 1),
                                 transpose=True, new_pos=(0, 3, 1, 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
help_ = "Load model checkpoints"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()


class Softmax(nn.Sequential):
    def __init__(self, num_features, num_classes=10):
        super(Softmax, self).__init__(
            nn.Linear(num_features, num_classes)
        )


# model = Softmax(imsize*imsize, 10).to(device)
model = LeNet5(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if args.weights:
    print('=> loading checkpoint %s' % args.weights)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint %s' % args.weights)
else:
    # train
    epochs = 20
    batch_size = 100
    epoch_steps = np.ceil(trainset.num_examples/batch_size).astype('int')
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

            # print statistics every 100 steps
            if (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}"
                      .format(epoch + 1, epochs,
                              (step + 1) * batch_size, trainset.num_examples,
                              loss.item()))

    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               'cnn_mnist_ckpt')

with torch.no_grad():
    loss_test = 0
    correct_test = 0
    while testset.epochs_completed <= 0:
        X_batch, y_batch = testset.next_batch(1000)
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)
        yp_batch = model(X_batch)
        loss_test += F.nll_loss(yp_batch, y_batch.long(), reduction='sum').item()
        # get the index of the max log-probability
        yp_batch = yp_batch.argmax(dim=1, keepdim=True)
        correct_test += yp_batch.eq(y_batch.view_as(yp_batch)).sum().item()

    score = [0, 0]
    score[0] = loss_test/testset.num_examples
    score[1] = correct_test/testset.num_examples
    print("Total loss on Testing Set:", score[0])
    print("Accuracy of Testing Set:", score[1])
