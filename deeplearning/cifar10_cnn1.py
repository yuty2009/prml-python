# -*- coding: utf-8 -*-

import argparse
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.cifarreader import *
from deeplearning.cnn.pytorch.cifarcnn import *

imsize = 32
datapath = 'e:/prmldata/cifar-10/python'
cifar = CIFARReader(datapath, num_classes = 10)
trainset = cifar.get_train_dataset(onehot_label=False,
                                   reshape=True, new_shape=(-1, 3, imsize, imsize))
testset = cifar.get_test_dataset(onehot_label=False,
                                 reshape=True, new_shape=(-1, 3, imsize, imsize))
label_names = cifar.get_label_names()

#Visualizing CIFAR 10
n = 10
figure = np.zeros((imsize * n, imsize * n, 3))
for j in range(n):
    for k in range(n):
        i = np.random.choice(range(len(trainset.images)))
        figure[j * imsize: (j + 1) * imsize,
        k * imsize: (k + 1) * imsize, :] = \
            trainset.images[i,:,:,:].transpose((1, 2, 0))
plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

parser = argparse.ArgumentParser()
help_ = "Load h5 model trained weights"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()

cnn = CIFARCNN(input_shape=(imsize, imsize, 3)).to(device)
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)

if args.weights:
    print('=> loading checkpoint %s' % args.weights)
    checkpoint = torch.load(args.weights)
    cnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint %s' % args.weights)
else:
    # train
    epochs = 50
    batch_size = 100
    epoch_steps = np.ceil(trainset.num_examples/batch_size).astype('int')
    for epoch in range(epochs):
        for step in range(epoch_steps):
            X_batch, y_batch = trainset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch, device=device)
            y_batch = torch.tensor(y_batch, device=device)

            yp_batch = cnn(X_batch)
            loss = F.nll_loss(yp_batch, y_batch.long(), reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics every 100 steps
            if (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}"
                      .format(epoch + 1, epochs,
                              (step + 1) * batch_size, trainset.num_examples,
                              loss.item()))

    torch.save({'state_dict': cnn.state_dict(), 'optimizer': optimizer.state_dict()},
               'cnn_mnist_ckpt')

with torch.no_grad():
    loss_test = 0
    correct_test = 0
    while testset.epochs_completed <= 0:
        X_batch, y_batch = testset.next_batch(1000)
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)
        yp_batch = cnn(X_batch)
        loss_test += F.nll_loss(yp_batch, y_batch.long(), reduction='sum').item()
        # get the index of the max log-probability
        yp_batch = yp_batch.argmax(dim=1, keepdim=True)
        correct_test += yp_batch.eq(y_batch.view_as(yp_batch)).sum().item()

    score = [0, 0]
    score[0] = loss_test/testset.num_examples
    score[1] = correct_test/testset.num_examples
    print("Total loss on Testing Set:", score[0])
    print("Accuracy of Testing Set:", score[1])
