# -*- coding: utf-8 -*-

import copy
import numpy as np
from .layer.conv import *
from .layer.pooling import *
from .layer.fullconnect import *
from .layer.recurrent import *
from collections import deque

class Network(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, inputs):
        activations = inputs
        for l in self.layers:
            activations = l.forward(activations)
        return activations

    def forward_train(self, inputs):
        As = deque([])
        activations = inputs
        for l in self.layers:
            activations = l.forward(activations)
            As.appendleft(activations)
        return As, activations

    def backward(self, delta, As):
        for l in reversed(self.layers):
            Al = As.popleft()
            delta = l.backward(delta, Al)

    def applygrad(self, lr):
        for l in self.layers:
            if l.learnable:
                l.applygrad(lr)

    def train_step(self, inputs, outputs, lr=1e-4):
        As, activations = self.forward_train(inputs)
        delta = self.loss.deriv(activations, outputs)
        self.backward(delta, As)
        self.applygrad(lr)

    def checkgrads(self, inputs, outputs):
        epsilon = 1e-6
        tolerence = 1e-8
        zs, act = self.forward_train(inputs)
        delta = self.loss.deriv(act, outputs)
        self.backward(delta, zs)
        for l in range(len(self.layers)):
            layer = self.layers[l]
            if isinstance(layer, FullConnect):
                for i in range(self.layers[l].weights.shape[0]):
                    for j in range(self.layers[l].weights.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].weights[i][j] += epsilon
                        net_m.layers[l].weights[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].weights_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
            elif isinstance(layer, Conv2D):
                for i in range(self.layers[l].weights.shape[0]):
                    for j in range(self.layers[l].weights.shape[1]):
                        for m in range(self.layers[l].weights.shape[2]):
                            for n in range(self.layers[l].weights.shape[3]):
                                net_p = copy.deepcopy(self)
                                net_m = copy.deepcopy(self)
                                net_p.layers[l].weights[i,j,m,n] += epsilon
                                net_m.layers[l].weights[i,j,m,n] -= epsilon
                                act_p = net_p.forward(inputs)
                                act_m = net_m.forward(inputs)
                                loss_p = net_p.loss.compute(act_p, outputs)
                                loss_m = net_m.loss.compute(act_m, outputs)
                                dWij = (loss_p - loss_m)/(2*epsilon)
                                e = abs(dWij - self.layers[l].weights_grads[i,j,m,n])
                                if e > tolerence:
                                    print('numerical gradient checking failed')
            elif isinstance(layer, RNNCell):
                for i in range(self.layers[l].wo.shape[0]):
                    for j in range(self.layers[l].wo.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].wo[i][j] += epsilon
                        net_m.layers[l].wo[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].wo_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].wh.shape[0]):
                    for j in range(self.layers[l].wh.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].wh[i][j] += epsilon
                        net_m.layers[l].wh[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].wh_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].wi.shape[0]):
                    for j in range(self.layers[l].wi.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].wi[i][j] += epsilon
                        net_m.layers[l].wi[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].wi_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
            elif isinstance(layer, LSTMCell):
                for i in range(self.layers[l].wo.shape[0]):
                    for j in range(self.layers[l].wo.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].wo[i][j] += epsilon
                        net_m.layers[l].wo[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].wo_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wi.shape[0]):
                    for j in range(self.layers[l].gates_wi.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wi[i][j] += epsilon
                        net_m.layers[l].gates_wi[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wi_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wf.shape[0]):
                    for j in range(self.layers[l].gates_wf.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wf[i][j] += epsilon
                        net_m.layers[l].gates_wf[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wf_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wo.shape[0]):
                    for j in range(self.layers[l].gates_wo.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wo[i][j] += epsilon
                        net_m.layers[l].gates_wo[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wo_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wg.shape[0]):
                    for j in range(self.layers[l].gates_wg.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wg[i][j] += epsilon
                        net_m.layers[l].gates_wg[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wg_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
            elif isinstance(layer, GRUCell):
                for i in range(self.layers[l].wo.shape[0]):
                    for j in range(self.layers[l].wo.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].wo[i][j] += epsilon
                        net_m.layers[l].wo[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].wo_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wz.shape[0]):
                    for j in range(self.layers[l].gates_wz.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wz[i][j] += epsilon
                        net_m.layers[l].gates_wz[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wz_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wr.shape[0]):
                    for j in range(self.layers[l].gates_wr.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wr[i][j] += epsilon
                        net_m.layers[l].gates_wr[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wr_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')
                for i in range(self.layers[l].gates_wg.shape[0]):
                    for j in range(self.layers[l].gates_wg.shape[1]):
                        net_p = copy.deepcopy(self)
                        net_m = copy.deepcopy(self)
                        net_p.layers[l].gates_wg[i][j] += epsilon
                        net_m.layers[l].gates_wg[i][j] -= epsilon
                        act_p = net_p.forward(inputs)
                        act_m = net_m.forward(inputs)
                        loss_p = net_p.loss.compute(act_p, outputs)
                        loss_m = net_m.loss.compute(act_m, outputs)
                        dWij = (loss_p - loss_m)/(2*epsilon)
                        e = abs(dWij - self.layers[l].gates_wg_grads[i][j])
                        if e > tolerence:
                            print('numerical gradient checking failed')