# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.utils import plot_model
from utils.mnistreader import *
from deeplearning.vae.vae_keras import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
# for MLP based VAE
# trainset = mnist.get_train_dataset()
# testset = mnist.get_test_dataset()
# show(np.reshape(trainset.images[0,:], (imsize, imsize)))
# for convolutional VAE
trainset = mnist.get_train_dataset(reshape=True, new_shape=(imsize, imsize, 1))
testset = mnist.get_test_dataset(reshape=True, new_shape=(imsize, imsize, 1))
X_train, y_train = trainset.images, trainset.labels
X_test, y_test = testset.images, testset.labels

parser = argparse.ArgumentParser()
help_ = "Load h5 model trained weights"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()

x_h, x_w = 28, 28
x_dim = 784
h_dim = 32
z_dim = 2
# model = VAE(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
model = CVAE(input_shape=(x_h, x_w, 1), h_dim=h_dim, z_dim=z_dim)
vae = model.model()
vae.compile(optimizer='adam')
vae.summary()

if args.weights:
    vae.load_weights(args.weights)
    plot_model(vae, to_file='cvae.png', show_shapes=True)
else:
    vae.fit(X_train, batch_size=128, epochs=5, shuffle=True)
    plot_model(vae, to_file='cvae.png', show_shapes=True)
    vae.save_weights('cvae_mnist.h5')

# show distribution of latent variables
encoder = model.encoder()
z_test = encoder.predict(X_test, batch_size=128)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:,0], z_test[:,1], c=y_test)
plt.colorbar()
plt.show()

# show generated results varied with the code
n = 10  # figure with nxn digits
imsize = 28
figure = np.zeros((imsize * n, imsize * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
decoder = model.decoder()
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(imsize, imsize)
        figure[i * imsize: (i + 1) * imsize,
        j * imsize: (j + 1) * imsize] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

# show generated results
n = 10  # figure with nxn digits
imsize = 28
indices = np.kron(np.random.randint(0, X_test.shape[0], n),
                  np.ones(n)).astype('int')
x1 = X_test[indices]
x2 = vae.predict(x1)
figure1 = np.zeros((imsize * n, imsize * n))
figure2 = np.zeros((imsize * n, imsize * n))
for i in range(n):
    for j in range(n):
        digit1 = x1[i*n+j].reshape(imsize, imsize)
        digit2 = x2[i*n+j].reshape(imsize, imsize)
        figure1[i * imsize: (i + 1) * imsize, j * imsize: (j + 1) * imsize] = digit1
        figure2[i * imsize: (i + 1) * imsize, j * imsize: (j + 1) * imsize] = digit2

plt.figure(figsize=(10, 10))
plt.imshow(figure1, cmap='Greys_r')
plt.show()
plt.figure(figsize=(10, 10))
plt.imshow(figure2, cmap='Greys_r')
plt.show()

