# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
from utils import *
from torchutils import train_epoch, evaluate


class SimpleModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.features = nn.Conv1d(1, 2, 32, padding='same')
        self.classifier = nn.Sequential(
            nn.Linear(n_features*2, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleTransform(object):
    def __call__(self, x):
        x = torch.Tensor(x)
        return torch.unsqueeze(x, dim=0)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, transforms=None):
        if transforms == None:
            self.xs = xs
        else:
            self.xs = [transforms(x) for x in xs]
        self.ys = torch.Tensor(ys).long()

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)


f1 = 35
f2 = 20
fs = 128
N = 200
T = 1 / fs
L = 128
t = np.arange(L) * T
f = np.arange(L / 2) / L
x1 = np.zeros((N, len(t)))
x2 = np.zeros((N, len(t)))
y1 = np.ones((N, 1))
y2 = np.zeros((N, 1))
np.random.seed(123456)
for i in range(N):
    x1[i, :] = np.sin(2*np.pi*f1*t + np.random.rand())
    x2[i, :] = np.sin(2*np.pi*f2*t + np.random.rand())
x1 = x1 + np.random.rand(*x1.shape)
x2 = x2 + np.random.rand(*x2.shape)

pxx1 = abs(np.fft.fft(x1))[:, :L//2]
pxx2 = abs(np.fft.fft(x2))[:, :L//2]

xs = np.vstack((x1, x2))
ys = np.vstack((y1, y2)).squeeze()

torch.manual_seed(123456)
tf = SimpleTransform()
dataset = SimpleDataset(xs, ys, tf)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.epochs = 20
args.batch_size = 20
args.num_workers = 16
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_folds = 10
for i in range(n_folds):
    trainset, testset = torch.utils.data.random_split(dataset, [360, 40])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = SimpleModel(L, 2).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)

    for epoch in range(1, args.epochs+1):
        train_loss, train_accu = train_epoch(
            trainloader, model, criterion, optimizer, epoch, args)
        test_loss, test_accu = evaluate(
            testloader, model, criterion, epoch, args)


ks = model.features.weight.cpu().detach().numpy()
k1 = ks[0, 0, :]
k2 = ks[1, 0, :]
pkk1 = abs(np.fft.fft(k1, L))[:L//2]
pkk2 = abs(np.fft.fft(k2, L))[:L//2]
f11, pkk11 = sig.freqz(k1, fs=fs)
f11, pkk22 = sig.freqz(k2, fs=fs)
pkk11 = abs(pkk11)
pkk22 = abs(pkk22)

plt.subplot(221)
plt.plot(t * fs, x1[0], 'k')
plt.plot(t * fs, x2[0], 'r')
plt.title('Raw signal with random phase')
plt.legend(['%d Hz'%f1, '%d Hz'%f2])
plt.xlabel('t (milliseconds)')
plt.ylabel('X(t)')

plt.subplot(222)
plt.plot(f * fs, 10 * np.log10(pxx1[0]), 'k')
plt.plot(f * fs, 10 * np.log10(pxx2[0]), 'r')
plt.title('Single-Sided Amplitude Spectrum of X(t)')
plt.legend(['%d Hz'%f1, '%d Hz'%f2])
plt.xlabel('f (Hz)')
plt.ylabel('10*log10|P1(f)|')

# plt.subplot(223)
# plt.plot(k1, 'k')
# plt.plot(k2, 'r')
# plt.title('filters')
# plt.legend(['kernel-1', 'kernel-2'])
plt.subplot(223)
plt.plot(f11, pkk11, 'k')
plt.plot(f11, pkk22, 'r')
plt.title('Freqz of filters')
plt.legend(['kernel-1', 'kernel-2'])
plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')

plt.subplot(224)
plt.plot(f * fs, pkk1, 'k')
plt.plot(f * fs, pkk2, 'r')
plt.title('FFT of filters')
plt.legend(['kernel-1', 'kernel-2'])
plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')

plt.tight_layout()
plt.show()