# -*- coding: utf-8 -*-
#
# reference:
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN(nn.Module):
    def __init__(self, image_shape, latent_dim, n_filters_g=16, n_filters_d=16):
        super(DCGAN, self).__init__()
        self.image_shape = image_shape
        # Initialize generator and discriminator
        self.generator = Generator(latent_dim, image_shape, n_filters_g)
        self.discriminator = Discriminator(image_shape, n_filters_d)
        # Loss function
        self.loss_adversarial = torch.nn.BCELoss()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=0.0002, betas=(0.5, 0.999))


class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape, n_filters_g):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = image_shape[-1] // 4
        self.out_channels = image_shape[0]
        self.n_filters_g = n_filters_g

        self.layer1 = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        def block(in_channels, out_channels, upsample=1,
                  act=nn.LeakyReLU(0.2, inplace=True), normalize=True):
            layers = [nn.Upsample(scale_factor=upsample),
                      nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act)
            return layers

        self.conv_blocks = nn.Sequential(
            *block(128,           n_filters_g*8, 2),
            *block(n_filters_g*8, n_filters_g*4, 2),
            nn.ConvTranspose2d(n_filters_g*4, self.out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z_img = self.layer1(z)
        z_img = z_img.view(z_img.size(0), 128, self.init_size, self.init_size)
        x_img = self.conv_blocks(z_img)
        return x_img


class Discriminator(nn.Module):
    def __init__(self, image_shape, n_filters_d):
        super(Discriminator, self).__init__()

        self.in_channels = image_shape[0]
        self.image_size = image_shape[-1]
        self.n_filters_d = n_filters_d

        def block(in_channels, out_channels, kernel_size,
                  stride=1, padding=0, bias=False,
                  act=nn.LeakyReLU(0.2, inplace=True), normalize=True):
            layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias), act, nn.Dropout2d(0.25)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            return layers

        self.model = nn.Sequential(
            *block(self.in_channels, n_filters_d * 1, 3, 1, 1, normalize=False),
            *block(n_filters_d * 1,  n_filters_d * 2, 3, 2, 1),
            *block(n_filters_d * 2,  n_filters_d * 4, 3, 1, 1),
            *block(n_filters_d * 4,  n_filters_d * 8, 3, 2, 1),
        )

        # The height and width of downsampled image
        ds_size = self.image_size // 4
        self.flatten = nn.Sequential(
            nn.Linear(n_filters_d * 8 * ds_size ** 2, 1),
            nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.flatten(out)
        return validity
