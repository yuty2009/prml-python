''' This module contains code to handle data '''

import os
import sys
import scipy
import numpy as np
import scipy.ndimage
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MNISTSet(Dataset):
    def __init__(
        self, root='resources/', subset='train', image_size=28, color=False, rescale=True,
    ):
        self.root = root
        self.subset = subset
        self.color = color
        self.rescale = rescale
        self.image_size = image_size
        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        # Load Lena image to memory
        self.lena = Image.open('selfsupervised/resources/lena.jpg') 

    def load_dataset(self):
        # Credit for this function: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images(self.root + '/train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels(self.root + '/train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images(self.root + '/t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels(self.root + '/t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def process_batch(self, batch, batch_size):

        # Resize from 28x28 to 64x64
        if self.image_size == 64:
            batch_resized = []
            for i in range(batch.shape[0]):
                # resize to 64x64 pixels
                batch_resized.append(scipy.ndimage.zoom(batch[i, :, :], 2.3, order=1))
            batch = np.stack(batch_resized)

        # Convert to RGB
        batch = batch.reshape((batch_size, 1, self.image_size, self.image_size))
        batch = np.concatenate([batch, batch, batch], axis=1)

        return batch
    
    def __getitem__(self, index):
        if self.subset == 'train':
            x, y = self.X_train, self.y_train
        elif self.subset == 'valid':
            x, y = self.X_test, self.y_test
        elif self.subset == 'test':
            x, y = self.X_val, self.y_val
        im = self.process_batch(x[index], 1)
        im = im[0]
        return im, y[index]

    def __len__(self):
        if self.subset == 'train':
            y_len = self.y_train.shape[0]
        elif self.subset == 'valid':
            y_len = self.y_val.shape[0]
        elif self.subset == 'test':
            y_len = self.y_test.shape[0]
        return y_len


class MNISTSeqSet(Dataset):
    def __init__(
        self, root='resources/', subset='train', image_size=28, color=False, rescale=True,
        terms = 4, predict_terms = 4,
    ):
        self.root = root
        self.subset = subset
        self.color = color
        self.rescale = rescale
        self.image_size = image_size
        self.terms = terms
        self.predict_terms = predict_terms
        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        # Load Lena image to memory
        self.lena = Image.open('selfsupervised/resources/lena.jpg') 

    def load_dataset(self):
        # Credit for this function: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images(self.root + '/train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels(self.root + '/train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images(self.root + '/t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels(self.root + '/t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_images_by_labels(self, subset, labels):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Find samples matching labels
        idxs = []
        for i, label in enumerate(labels):

            idx = np.where(y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idxs.append(idx_sel)

        # Retrieve images
        images = X[np.array(idxs), 0, :].reshape((len(labels), 28, 28))

        # Process images
        images = self.process_batch(images, len(labels))

        return images.astype('float32'), labels.astype('int32')

    def process_batch(self, batch, batch_size):

        # Resize from 28x28 to 64x64
        if self.image_size == 64:
            batch_resized = []
            for i in range(batch.shape[0]):
                # resize to 64x64 pixels
                batch_resized.append(scipy.ndimage.zoom(batch[i, :, :], 2.3, order=1))
            batch = np.stack(batch_resized)

        # Convert to RGB
        batch = batch.reshape((batch_size, 1, self.image_size, self.image_size))
        batch = np.concatenate([batch, batch, batch], axis=1)

        # Modify images if color distribution requested
        if self.color:

            # Binarize images
            batch[batch >= 0.5] = 1
            batch[batch < 0.5] = 0

            # For each image in the mini batch
            for i in range(batch_size):

                # Take a random crop of the Lena image (background)
                x_c = np.random.randint(0, self.lena.size[0] - self.image_size)
                y_c = np.random.randint(0, self.lena.size[1] - self.image_size)
                image = self.lena.crop((x_c, y_c, x_c + self.image_size, y_c + self.image_size))
                image = np.asarray(image).transpose((2, 0, 1)) / 255.0

                # Randomly alter the color distribution of the crop
                for j in range(3):
                    image[j, :, :] = (image[j, :, :] + np.random.uniform(0, 1)) / 2.0

                # Invert the color of pixels where there is a number
                image[batch[i, :, :, :] == 1] = 1 - image[batch[i, :, :, :] == 1]
                batch[i, :, :, :] = image

        # Rescale to range [-1, +1]
        if self.rescale:
            batch = batch * 2 - 1

        return batch
    
    def __getitem__(self, index):
        # Set ordered predictions for positive samples
        seed = np.random.randint(0, 10)
        sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), 10)
        sentence_label = np.random.randint(2)
        if sentence_label <= 0:
            # Set random predictions for negative sample
            # Each predicted term draws a number from a distribution that excludes itself
            numbers = np.arange(0, 10)
            predicted_terms = sentence[-self.predict_terms:]
            for i, p in enumerate(predicted_terms):
                predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
            sentence[-self.predict_terms:] = np.mod(predicted_terms, 10)

        # Retrieve actual images
        images, _ = self.get_images_by_labels(self.subset, sentence)

        x_images = images[:-self.predict_terms, ...]
        y_images = images[-self.predict_terms:, ...]

        return [x_images, y_images], sentence_label

    def __len__(self):
        if self.subset == 'train':
            y_len = self.y_train.shape[0]
        elif self.subset == 'valid':
            y_len = self.y_val.shape[0]
        elif self.subset == 'test':
            y_len = self.y_test.shape[0]
        return y_len


def plot_imagegrid(x, output_path=None):
    images = x
    n_batches = images.shape[0]
    n_cols = 8
    n_rows = int(np.ceil(n_batches / n_cols))
    plt.figure()
    for n_b in range(n_batches):
        im = images[n_b, :, :, :]
        im = np.transpose(im, (1, 2, 0))
        plt.subplot(n_rows, n_cols, n_b + 1)
        plt.imshow(im)
        plt.axis('off')

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


def plot_sequences(x, y, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    plt.figure()
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            im = images[n_b, n_t, :, :, :]
            im = np.transpose(im, (1, 2, 0))
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(im)
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b])

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


if __name__ == "__main__":

    trainset = MNISTSeqSet(
        root='/Users/yuty2009/data/prmldata/mnist/MNIST/raw', subset='train',
        image_size=64, color=True, rescale=False, terms=4, predict_terms=4,
    )
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
    for (x, y), labels in trainloader:
        plot_sequences(x, y, labels, output_path=r'selfsupervised/resources/batch_sample_sorted.png')
        break

    memoryset = MNISTSet(
        root='/Users/yuty2009/data/prmldata/mnist/MNIST/raw', subset='train',
        image_size=64, color=True, rescale=False,
    )
    memoryloader = DataLoader(memoryset, batch_size=64, shuffle=True)
    for x, labels in memoryloader:
        plot_imagegrid(x)
        break


