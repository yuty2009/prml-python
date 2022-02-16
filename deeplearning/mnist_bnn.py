# -*- coding: utf-8 -*-

import argparse
import torch
import torch.optim as optim
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataloader.mnistreader import *
from deeplearning.bnn.bmlp import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
trainset = mnist.get_train_dataset()
testset = mnist.get_test_dataset()
# show(np.reshape(trainset.images[0,:], (imsize, imsize)))

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser()
help_ = "Load model checkpoints"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()

x_dim = 784
h_dims = [128, 84, 10]

mlp = BayesMLP(input_dim=x_dim, hidden_dims=h_dims,
               acts=[nn.ReLU(inplace=True),
                     nn.ReLU(inplace=True),
                     None],
               priors=[GaussPrior(mu=0, sigma=0.5),
                       GaussPrior(mu=0, sigma=0.5),
                       GaussPrior(mu=0, sigma=0.5)]).to(device)

optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

if args.weights:
    print('=> loading checkpoint %s' % args.weights)
    checkpoint = torch.load(args.weights)
    mlp.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint %s' % args.weights)
else:
    # train
    n_samples = 5
    epochs = 10
    batch_size = 100
    num_batches = np.ceil(trainset.num_examples / batch_size).astype('int')
    weight_kl = 1.0 / num_batches
    for epoch in range(epochs):
        for step in range(num_batches):
            X_batch, y_batch = trainset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch, device=device)
            y_batch = torch.tensor(y_batch, device=device)

            yp_batch, loss_kl = mlp(X_batch, sample=True)
            loss_ce = F.cross_entropy(yp_batch, y_batch.long(), reduction='sum')
            loss = loss_ce * trainset.num_examples + weight_kl * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics every 100 steps
            if (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], CrossEntropy Loss: {:.4f}, "
                      "KL Div: {:.4f} Total loss {:.4f}"
                      .format(epoch + 1, epochs,
                              (step + 1) * batch_size, trainset.num_examples,
                              loss_ce.item(), loss_kl.item(), loss.item()))

    torch.save({'state_dict': mlp.state_dict(), 'optimizer': optimizer.state_dict()},
               'mnist_bnn_ckpt')

with torch.no_grad():

    loss_test = 0
    correct_test = 0
    while testset.epochs_completed <= 0:
        X_batch, y_batch = testset.next_batch(1000)
        X_batch = torch.tensor(X_batch, device=device)
        y_batch = torch.tensor(y_batch, device=device)
        yp_batch, _ = mlp(X_batch)
        # yp_batch, _ = mlp.predict_mcmc(X_batch, n_samples=100)
        loss_test += F.cross_entropy(yp_batch, y_batch.long(), reduction='sum').item()
        # get the index of the max log-probability
        yp_batch = yp_batch.argmax(dim=1, keepdim=True)
        correct_test += yp_batch.eq(y_batch.view_as(yp_batch)).sum().item()

    score = [0, 0]
    score[0] = loss_test/testset.num_examples
    score[1] = correct_test/testset.num_examples
    print("Total loss on Testing Set:", score[0])
    print("Accuracy of Testing Set:", score[1])