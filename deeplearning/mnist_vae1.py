# -*- coding: utf-8 -*-

import argparse
import torch
import torch.optim as optim
from scipy.stats import norm
import matplotlib.pyplot as plt
from utils.mnistreader import *
from deeplearning.vae.vae_pytorch import *

f_train_images = 'e:/prmldata/mnist/train-images-idx3-ubyte'
f_train_labels = 'e:/prmldata/mnist/train-labels-idx1-ubyte'
f_test_images = 'e:/prmldata/mnist/t10k-images-idx3-ubyte'
f_test_labels = 'e:/prmldata/mnist/t10k-labels-idx1-ubyte'

imsize = 28
mnist = MNISTReader(f_train_images, f_train_labels, f_test_images, f_test_labels)
# for MLP based VAE
trainset = mnist.get_train_dataset()
testset = mnist.get_test_dataset()
# show(np.reshape(trainset.images[0,:], (imsize, imsize)))
# for convolutional VAE
# trainset = mnist.get_train_dataset(onehot_label=True,
#                                    reshape=True, new_shape=(-1, imsize, imsize, 1),
#                                    tranpose=True, new_pos=(0, 3, 1, 2))
# testset = mnist.get_test_dataset(onehot_label=True,
#                                  reshape=True, new_shape=(-1, imsize, imsize, 1),
#                                  tranpose=True, new_pos=(0, 3, 1, 2))

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

parser = argparse.ArgumentParser()
help_ = "Load h5 model trained weights"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()

x_dim = 784
h_dim = 32
z_dim = 2
vae = VAE(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim).to(device)
# vae = CVAE(input_shape=(imsize, imsize, 1), h_dim=h_dim, z_dim=z_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

if args.weights:
    print('=> loading checkpoint %s' % args.weights)
    checkpoint = torch.load(args.weights)
    vae.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint %s' % args.weights)
else:
    # train
    epochs = 50
    batch_size = 100
    epoch_steps = np.ceil(trainset.num_examples / batch_size).astype('int')
    for epoch in range(epochs):
        for step in range(epoch_steps):
            X_batch, y_batch = trainset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch).to(device)

            X_decoded, z_mean, z_log_var = vae(X_batch)
            loss, loss_recon, loss_kl = vae.loss(X_batch, X_decoded, z_mean, z_log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics every 100 steps
            if (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Recon Loss: {:.4f}, "
                      "KL Div: {:.4f} Total loss {:.4f}"
                      .format(epoch + 1, epochs,
                              (step + 1) * batch_size, trainset.num_examples,
                              loss_recon.item(), loss_kl.item(), loss.item()))

    torch.save({'state_dict': vae.state_dict(), 'optimizer': optimizer.state_dict()},
               'vae_mnist_ckpt')

# test
with torch.no_grad():
    # show distribution of latent variables
    x_test = torch.tensor(testset.images).to(device)
    z_test, _ = vae.encode(x_test)
    z_test = z_test.cpu().data.numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(z_test[:, 0], z_test[:, 1], c=testset.labels)
    plt.colorbar()
    plt.show()

    # show generated results varied with the code
    n = 10  # figure with nxn digits
    imsize = 28
    figure = np.zeros((imsize * n, imsize * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]).astype('float32')
            z_sample = torch.tensor(z_sample).to(device)
            x_decoded = vae.decode(z_sample)
            x_decoded = x_decoded.cpu().data.numpy()
            digit = x_decoded[0].reshape(imsize, imsize)
            figure[i * imsize: (i + 1) * imsize,
            j * imsize: (j + 1) * imsize] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

    # show generated results
    n = 10  # figure with nxn digits
    imsize = 28
    indices = np.kron(np.random.randint(0, testset.num_examples, n),
                      np.ones(n)).astype('int')
    x1 = torch.tensor(testset.images[indices]).to(device)
    x2 = vae(x1)[0]
    x1 = x1.cpu().data.numpy()
    x2 = x2.cpu().data.numpy()
    figure1 = np.zeros((imsize * n, imsize * n))
    figure2 = np.zeros((imsize * n, imsize * n))
    for i in range(n):
        for j in range(n):
            digit1 = x1[i * n + j].reshape(imsize, imsize)
            digit2 = x2[i * n + j].reshape(imsize, imsize)
            figure1[i * imsize: (i + 1) * imsize, j * imsize: (j + 1) * imsize] = digit1
            figure2[i * imsize: (i + 1) * imsize, j * imsize: (j + 1) * imsize] = digit2

    plt.figure(figsize=(10, 10))
    plt.imshow(figure1, cmap='Greys_r')
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(figure2, cmap='Greys_r')
    plt.show()
