# -*- coding: utf-8 -*-
'''
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
[2] https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K


class VAE:
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        # encoder
        self.x_inputs = Input(shape=(x_dim,))
        self.hidden1 = Dense(h_dim, activation='relu')(self.x_inputs)
        self.z_mean = Dense(z_dim)(self.hidden1)
        self.z_log_var = Dense(z_dim)(self.hidden1)
        self.z_sampled = Lambda(self.reparameterize,
                           output_shape=(z_dim,))([self.z_mean, self.z_log_var])
        # decoder
        self.z_inputs = Input(shape=(z_dim,))
        self.hidden2 = Dense(h_dim, activation='relu')(self.z_inputs)
        self.x_decoded = Dense(x_dim, activation='sigmoid')(self.hidden2)

    def model(self):
        x_outputs = self.decoder()(self.z_sampled)
        vae = Model(self.x_inputs, x_outputs)
        loss_recon = K.sum(K.binary_crossentropy(self.x_inputs, x_outputs), axis=-1)
        loss_kl = - 0.5 * K.sum(1 + self.z_log_var
                                - K.square(self.z_mean)
                                - K.exp(self.z_log_var), axis=-1)
        loss_vae = K.mean(loss_recon + loss_kl)
        vae.add_loss(loss_vae)
        return vae

    def encoder(self):
        return Model(self.x_inputs, self.z_mean)

    def reparameterize(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + epsilon * K.exp(0.5 * z_log_var)

    def decoder(self):
        return Model(self.z_inputs, self.x_decoded)


class CVAE():
    def __init__(self, input_shape, h_dim, z_dim):
        super(CVAE, self).__init__()
        # encoder
        self.x_inputs = Input(shape=input_shape)
        self.conv1 = Conv2D(32, kernel_size=3, activation='relu',
                            strides=2, padding='same')(self.x_inputs)
        self.conv2 = Conv2D(64, kernel_size=3, activation='relu',
                            strides=2, padding='same')(self.conv1)
        shape = K.int_shape(self.conv2)
        self.flatten = Flatten()(self.conv2)
        self.hidden1 = Dense(h_dim, activation='relu')(self.flatten)
        self.z_mean = Dense(z_dim)(self.hidden1)
        self.z_log_var = Dense(z_dim)(self.hidden1)
        self.z_sampled = Lambda(self.reparameterize,
                                output_shape=(z_dim,))([self.z_mean, self.z_log_var])
        # decoder
        self.z_inputs = Input(shape=(z_dim,))
        self.hidden2 = Dense(shape[1]*shape[2]*shape[3], activation='relu')(self.z_inputs)
        self.unflatten = Reshape((shape[1], shape[2], shape[3]))(self.hidden2)
        self.deconv1 = Conv2DTranspose(64, kernel_size=3, activation='relu',
                                       strides=2, padding='same')(self.unflatten)
        self.deconv2 = Conv2DTranspose(32, kernel_size=3, activation='relu',
                                       strides=2, padding='same')(self.deconv1)
        self.x_decoded = Conv2DTranspose(1, kernel_size=3, activation='sigmoid',
                                         padding='same')(self.deconv2)

    def model(self):
        x_outputs = self.decoder()(self.z_sampled)
        vae = Model(self.x_inputs, x_outputs)
        loss_recon = K.sum(K.binary_crossentropy(self.x_inputs, x_outputs),
                           axis=[1, 2, 3]) # different from VAE
        loss_kl = - 0.5 * K.sum(1 + self.z_log_var
                                - K.square(self.z_mean)
                                - K.exp(self.z_log_var), axis=-1)
        loss_vae = K.mean(loss_recon + loss_kl)
        vae.add_loss(loss_vae)
        return vae

    def encoder(self):
        return Model(self.x_inputs, self.z_mean)

    def reparameterize(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + epsilon * K.exp(0.5 * z_log_var)

    def decoder(self):
        return Model(self.z_inputs, self.x_decoded)