# -*- coding: utf-8 -*-
#
# reference:
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN(nn.Module):
    def __init__(self, image_shape, latent_dim):
        super(GAN, self).__init__()
        self.image_shape = image_shape
        # Initialize generator and discriminator
        self.generator = Generator(output_shape=image_shape,
                                   input_dim=latent_dim,
                                   hidden_dims=[128, 256, 512, 1024, np.prod(image_shape)],
                                   acts=[nn.LeakyReLU(0.2, inplace=True),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Tanh()])
        self.discriminator = Discriminator(input_shape=image_shape,
                                           hidden_dims=[512, 256, 1],
                                           acts=[nn.LeakyReLU(0.2, inplace=True),
                                                 nn.LeakyReLU(0.2, inplace=True),
                                                 nn.Sigmoid()])
        # Loss function
        self.loss_adversarial = torch.nn.BCELoss()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))


class Generator(nn.Module):
    def __init__(self, output_shape, input_dim, hidden_dims, acts):
        super(Generator, self).__init__()

        self.output_shape = output_shape
        self.input_dim = input_dim
        self.n_hid = len(hidden_dims)

        if acts is None:
            self.acts = []
            for i in range(self.n_hid - 1):
                self.acts.append(nn.LeakyReLU(0.2, inplace=True))
            self.acts.append(None)
        elif isinstance(acts, list) or isinstance(acts, tuple):
            self.acts = acts
        else:
            self.acts = []
            for i in range(self.n_hid):
                self.acts.append(acts)

        def block(n_in, n_out, act, normalize=True):
            layers = [nn.Linear(n_in, n_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_out, 0.8))
            layers.append(act)
            return layers

        in_dims = [input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.layers = []
        for i in range(self.n_hid):
            if i is 0:
                self.layers = self.layers + \
                                  block(in_dims[i], out_dims[i],acts[i], normalize=False)
            else:
                self.layers = self.layers + \
                                  block(in_dims[i], out_dims[i], acts[i])
        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.output_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dims, acts):
        super(Discriminator, self).__init__()

        self.input_dim = np.prod(input_shape)
        self.n_hid = len(hidden_dims)

        if acts is None:
            self.acts = []
            for i in range(self.n_hid - 1):
                self.acts.append(nn.LeakyReLU(0.2, inplace=True))
            self.acts.append(None)
        elif isinstance(acts, list) or isinstance(acts, tuple):
            self.acts = acts
        else:
            self.acts = []
            for i in range(self.n_hid):
                self.acts.append(acts)

        def block(n_in, n_out, act, normalize=True):
            layers = [nn.Linear(n_in, n_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_out, 0.8))
            layers.append(act)
            return layers

        in_dims = [self.input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.layers = []
        for i in range(self.n_hid):
            self.layers = self.layers + \
                          block(in_dims[i], out_dims[i], acts[i], normalize=False)
        self.model = nn.Sequential(*self.layers)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
