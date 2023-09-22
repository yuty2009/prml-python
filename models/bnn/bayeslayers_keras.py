# -*- coding: utf-8 -*-
#
# reference: http://krasserm.github.io/2019/03/14/bayesian-neural-networks/

from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
# import tensorflow_probability as tfp
from .priors import *


class BayesLinear(Layer):
    def __init__(self, units, prior=None, activation=None, kl_weight=1, **kwargs):
        self.units = units
        self.prior = prior
        self.activation = activations.get(activation)
        self.kl_weight = kl_weight
        self.init_sigma = 0.5

        super(BayesLinear, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.normal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.normal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super(BayesLinear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = K.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * K.random_normal(self.kernel_mu.shape)

        bias_sigma = K.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * K.random_normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = GaussPrior(mu, sigma, backend=K)
        # variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) -
                                      self.prior.log_prob(w))


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = GaussPrior(mu=y_pred, sigma=sigma, backend=K)
    # dist = tfp.distributions.Normal(y_pred, sigma)
    return K.sum(-dist.log_prob(y_obs))
