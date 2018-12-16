# -*- coding: utf-8 -*-
# reference: https://zhuanlan.zhihu.com/p/25858226?utm_source=qq&utm_medium=social&utm_oi=875081993430380544

from .base import *
from ..activation import *

class RNNCell(object):
    def __init__(self, inshape, nsteps, hdim, outdim, activation=Softmax(), hactivation=Sigmoid(),
                 hinit=None, weight_init=weight_init, bias_init=bias_init):
        self.learnable = True
        self.inshape = inshape
        self.batchsize = inshape[0]
        self.nsteps = nsteps
        self.hdim = hdim  # [h_Nodes]
        self.outdim = outdim  # [out_Nodes]
        self.outshape = [self.batchsize, nsteps, outdim]
        self.activation = activation
        self.hactivation = hactivation
        self.hinit = hinit
        if self.hinit is None:
            self.hinit = np.zeros([self.batchsize, self.hdim])
        self.wi = weight_init([self.inshape[-1], self.hdim])
        self.wh = weight_init([self.hdim, self.hdim])
        self.bh = bias_init(self.hdim)
        self.wo = weight_init([self.hdim, self.outdim])
        self.bo = bias_init(self.outdim)
        self.wi_grads = np.zeros(self.wi.shape)
        self.wh_grads = np.zeros(self.wh.shape)
        self.bh_grads = np.zeros(self.bh.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)

    def onestep(self, H_prev, X_current):
        H_next = self.hactivation.compute(np.dot(X_current, self.wi) + np.dot(H_prev, self.wh) + self.bh)
        return H_next

    def forward(self, X):
        self.X = X.reshape([self.batchsize, self.nsteps, -1])
        Os = np.zeros([self.batchsize, self.nsteps, self.outdim])
        self.Hs = np.zeros([self.batchsize, self.nsteps+1, self.hdim])
        self.Hs[:,-1,:] = self.hinit
        for t in range(self.nsteps):
            self.Hs[:,t,:] = self.onestep(self.Hs[:,t-1,:], self.X[:,t,:])
            Os[:,t,:] = self.activation.compute(np.dot(self.Hs[:,t,:], self.wo) + self.bo)
        return Os

    def backward(self, delta, A):
        delta *= self.activation.deriv(A)
        next_delta = np.zeros([self.batchsize, self.nsteps, self.inshape[-1]])
        # For each output backwards...
        for t in np.arange(self.nsteps)[::-1]:
            # delta: NxTxC, which is the same as the output
            self.wo_grads += np.dot(self.Hs[:,t,:].T, delta[:,t,:]) / self.batchsize
            self.bo_grads += np.mean(delta[:,t,:], axis=0)
            delta_t = np.dot(delta[:,t,:], self.wo.T)
            # Backpropagation through time
            for bptt_step in np.arange(t+1)[::-1]:
                delta_t *= self.hactivation.deriv(self.Hs[:, bptt_step, :])
                self.wh_grads += np.dot(self.Hs[:,bptt_step-1,:].T, delta_t) / self.batchsize
                self.wi_grads += np.dot(self.X[:,bptt_step,:].T, delta_t) / self.batchsize
                self.bh_grads += np.mean(delta_t, axis=0)
                next_delta[:, bptt_step, :] += np.dot(delta_t, self.wi.T)
                delta_t = np.dot(delta_t, self.wh.T)
        return next_delta

    def applygrad(self, lr=1e-4, wd=4e-4):
        self.wi *= (1. - wd)
        self.wh *= (1. - wd)
        self.bh *= (1. - wd)
        self.wo *= (1. - wd)
        self.bo *= (1. - wd)
        self.wi -= lr * self.wi_grads
        self.wh -= lr * self.wh_grads
        self.bh -= lr * self.bh_grads
        self.wo -= lr * self.wo_grads
        self.bo -= lr * self.bo_grads
        self.wi_grads = np.zeros(self.wi.shape)
        self.wh_grads = np.zeros(self.wh.shape)
        self.bh_grads = np.zeros(self.bh.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)


class LSTMCell(object):
    def __init__(self, inshape, nsteps, hdim, outdim, activation=Softmax(), hactivation=Tanh(),
                 hinit=None, cinit=None, weight_init=weight_init, bias_init=bias_init):
        self.learnable = True
        self.inshape = inshape
        self.batchsize = inshape[0]
        self.nsteps = nsteps
        self.hdim = hdim  # [h_Nodes]
        self.outdim = outdim  # [out_Nodes]
        self.outshape = [self.batchsize, nsteps, outdim]
        self.activation = activation
        self.hactivation = hactivation
        self.hinit = hinit
        if self.hinit is None:
            self.hinit = np.zeros([self.batchsize, self.hdim])
        self.cinit = cinit
        if self.cinit is None:
            self.cinit = np.zeros([self.batchsize, self.hdim])
        xhlen = self.inshape[-1] + self.hdim
        self.gates_wi = weight_init([xhlen, self.hdim])
        self.gates_wf = weight_init([xhlen, self.hdim])
        self.gates_wo = weight_init([xhlen, self.hdim])
        self.gates_wg = weight_init([xhlen, self.hdim])
        self.gates_bi = bias_init(self.hdim)
        self.gates_bf = bias_init(self.hdim)
        self.gates_bo = bias_init(self.hdim)
        self.gates_bg = bias_init(self.hdim)
        self.wo = weight_init([self.hdim, self.outdim])
        self.bo = bias_init(self.outdim)
        self.gates_wi_grads = np.zeros(self.gates_wi.shape)
        self.gates_wf_grads = np.zeros(self.gates_wf.shape)
        self.gates_wo_grads = np.zeros(self.gates_wo.shape)
        self.gates_wg_grads = np.zeros(self.gates_wg.shape)
        self.gates_bi_grads = np.zeros(self.gates_bi.shape)
        self.gates_bf_grads = np.zeros(self.gates_bf.shape)
        self.gates_bo_grads = np.zeros(self.gates_bo.shape)
        self.gates_bg_grads = np.zeros(self.gates_bg.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)

    def onestep(self, H_prev, C_prev, X_current):
        XH = np.concatenate((X_current, H_prev), axis=1)
        i = Sigmoid().compute(np.dot(XH, self.gates_wi) + self.gates_bi)
        f = Sigmoid().compute(np.dot(XH, self.gates_wf) + self.gates_bf)
        o = Sigmoid().compute(np.dot(XH, self.gates_wo) + self.gates_bo)
        g = Tanh().compute(np.dot(XH, self.gates_wg) + self.gates_bg)
        C_next = f * C_prev + i * g
        H_next = o * C_next # o * Tanh().compute(C_next)
        return H_next, C_next, i, f, o, g

    def forward(self, X):
        self.X = X.reshape([self.batchsize, self.nsteps, -1])
        Os = np.zeros([self.batchsize, self.nsteps, self.outdim])
        self.Hs = np.zeros([self.batchsize, self.nsteps + 1, self.hdim])
        self.Cs = np.zeros([self.batchsize, self.nsteps + 1, self.hdim])
        self.states_i = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.states_f = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.states_o = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.states_g = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.Hs[:, -1, :] = self.hinit
        self.Cs[:, -1, :] = self.cinit
        for t in range(self.nsteps):
            self.Hs[:,t,:], self.Cs[:,t,:], \
            self.states_i[:,t,:], self.states_f[:,t,:], self.states_o[:,t,:], self.states_g[:,t,:]= \
                self.onestep(self.Hs[:,t-1,:], self.Cs[:,t-1,:], self.X[:,t,:])
            Os[:, t, :] = self.activation.compute(np.dot(self.Hs[:, t, :], self.wo) + self.bo)
        return Os

    def backward(self, delta, A):
        delta *= self.activation.deriv(A)
        next_delta = np.zeros([self.batchsize, self.nsteps, self.inshape[-1]])
        # For each output backwards...
        for t in np.arange(self.nsteps)[::-1]:
            # delta: NxTxC, which is the same as the output
            self.wo_grads += np.dot(self.Hs[:, t, :].T, delta[:, t, :]) / self.batchsize
            self.bo_grads += np.mean(delta[:, t, :], axis=0)
            delta_h = np.dot(delta[:, t, :], self.wo.T)
            delta_c = np.zeros([self.batchsize, self.hdim])
            # Backpropagation through time
            for bptt_step in np.arange(t + 1)[::-1]:
                XH = np.concatenate((self.X[:, bptt_step, :], self.Hs[:, bptt_step-1, :]), axis=1)
                # delta_c ???
                dc = self.states_o[:,bptt_step,:] * delta_h + delta_c
                di = self.states_g[:,bptt_step,:] * dc
                df = self.Cs[:, bptt_step-1, :] * dc
                do = self.Cs[:, bptt_step, :] * delta_h
                dg = self.states_i[:,bptt_step,:] * dc
                # diffs w.r.t. vector inside sigma / tanh function
                di_input = Sigmoid().deriv(self.states_i[:,bptt_step,:]) * di
                df_input = Sigmoid().deriv(self.states_f[:,bptt_step,:]) * df
                do_input = Sigmoid().deriv(self.states_o[:,bptt_step,:]) * do
                dg_input = Tanh().deriv(self.states_g[:,bptt_step,:]) * dg
                # diffs w.r.t. inputs
                self.gates_wi_grads += np.dot(XH.T, di_input) / self.batchsize
                self.gates_wf_grads += np.dot(XH.T, df_input) / self.batchsize
                self.gates_wo_grads += np.dot(XH.T, do_input) / self.batchsize
                self.gates_wg_grads += np.dot(XH.T, dg_input) / self.batchsize
                self.gates_bi_grads += np.mean(di_input)
                self.gates_bf_grads += np.mean(df_input)
                self.gates_bo_grads += np.mean(do_input)
                self.gates_bg_grads += np.mean(dg_input)
                # compute bottom diff
                dxc = np.zeros(XH.shape)
                dxc += np.dot(di_input, self.gates_wi.T)
                dxc += np.dot(df_input, self.gates_wf.T)
                dxc += np.dot(do_input, self.gates_wo.T)
                dxc += np.dot(dg_input, self.gates_wg.T)
                # save bottom diffs
                delta_c = dc * self.states_f[:,bptt_step,:]
                delta_h = dxc[:,self.inshape[-1]:]
                next_delta[:, bptt_step, :] += dxc[:,:self.inshape[-1]]
        return next_delta

    def applygrad(self, lr=1e-4, wd=4e-4):
        self.gates_wi *= (1. - wd)
        self.gates_wf *= (1. - wd)
        self.gates_wo *= (1. - wd)
        self.gates_wg *= (1. - wd)
        self.gates_bi *= (1. - wd)
        self.gates_bf *= (1. - wd)
        self.gates_bo *= (1. - wd)
        self.gates_bg *= (1. - wd)
        self.wo *= (1. - wd)
        self.bo *= (1. - wd)
        self.gates_wi -= lr * self.gates_wi_grads
        self.gates_wf -= lr * self.gates_wf_grads
        self.gates_wo -= lr * self.gates_wo_grads
        self.gates_wg -= lr * self.gates_wg_grads
        self.gates_bi -= lr * self.gates_bi_grads
        self.gates_bf -= lr * self.gates_bf_grads
        self.gates_bo -= lr * self.gates_bo_grads
        self.gates_bg -= lr * self.gates_bg_grads
        self.wo -= lr * self.wo_grads
        self.bo -= lr * self.bo_grads
        self.gates_wi_grads = np.zeros(self.gates_wi.shape)
        self.gates_wf_grads = np.zeros(self.gates_wf.shape)
        self.gates_wo_grads = np.zeros(self.gates_wo.shape)
        self.gates_wg_grads = np.zeros(self.gates_wg.shape)
        self.gates_bi_grads = np.zeros(self.gates_bi.shape)
        self.gates_bf_grads = np.zeros(self.gates_bf.shape)
        self.gates_bo_grads = np.zeros(self.gates_bo.shape)
        self.gates_bg_grads = np.zeros(self.gates_bg.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)


class GRUCell(object):
    def __init__(self, inshape, nsteps, hdim, outdim, activation=Softmax(), hactivation=Tanh(),
                 hinit=None, cinit=None, weight_init=weight_init, bias_init=bias_init):
        self.learnable = True
        self.inshape = inshape
        self.batchsize = inshape[0]
        self.nsteps = nsteps
        self.hdim = hdim  # [h_Nodes]
        self.outdim = outdim  # [out_Nodes]
        self.outshape = [self.batchsize, nsteps, outdim]
        self.activation = activation
        self.hactivation = hactivation
        self.hinit = hinit
        if self.hinit is None:
            self.hinit = np.zeros([self.batchsize, self.hdim])
        self.cinit = cinit
        if self.cinit is None:
            self.cinit = np.zeros([self.batchsize, self.hdim])
        xhlen = self.inshape[-1] + self.hdim
        self.gates_wz = weight_init([xhlen, self.hdim])
        self.gates_wr = weight_init([xhlen, self.hdim])
        self.gates_wg = weight_init([xhlen, self.hdim])
        self.gates_bz = bias_init(self.hdim)
        self.gates_br = bias_init(self.hdim)
        self.gates_bg = bias_init(self.hdim)
        self.wo = weight_init([self.hdim, self.outdim])
        self.bo = bias_init(self.outdim)
        self.gates_wz_grads = np.zeros(self.gates_wz.shape)
        self.gates_wr_grads = np.zeros(self.gates_wr.shape)
        self.gates_wg_grads = np.zeros(self.gates_wg.shape)
        self.gates_bz_grads = np.zeros(self.gates_bz.shape)
        self.gates_br_grads = np.zeros(self.gates_br.shape)
        self.gates_bg_grads = np.zeros(self.gates_bg.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)

    def onestep(self, H_prev, X_current):
        XH = np.concatenate((X_current, H_prev), axis=1)
        z = Sigmoid().compute(np.dot(XH, self.gates_wz) + self.gates_bz)
        r = Sigmoid().compute(np.dot(XH, self.gates_wr) + self.gates_br)
        XHP = np.concatenate((X_current, r * H_prev), axis=1)
        g = Tanh().compute(np.dot(XHP, self.gates_wg) + self.gates_bg)
        H_next = (1-z) * H_prev + z * g
        return H_next, z, r, g

    def forward(self, X):
        self.X = X.reshape([self.batchsize, self.nsteps, -1])
        Os = np.zeros([self.batchsize, self.nsteps, self.outdim])
        self.Hs = np.zeros([self.batchsize, self.nsteps + 1, self.hdim])
        self.states_z = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.states_r = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.states_g = np.zeros([self.batchsize, self.nsteps, self.hdim])
        self.Hs[:, -1, :] = self.hinit
        for t in range(self.nsteps):
            self.Hs[:,t,:], self.states_z[:,t,:], self.states_r[:,t,:], self.states_g[:,t,:]= \
                self.onestep(self.Hs[:,t-1,:], self.X[:,t,:])
            Os[:, t, :] = self.activation.compute(np.dot(self.Hs[:, t, :], self.wo) + self.bo)
        return Os

    def backward(self, delta, A):
        delta *= self.activation.deriv(A)
        next_delta = np.zeros([self.batchsize, self.nsteps, self.inshape[-1]])
        # For each output backwards...
        for t in np.arange(self.nsteps)[::-1]:
            # delta: NxTxC, which is the same as the output
            self.wo_grads += np.dot(self.Hs[:, t, :].T, delta[:, t, :]) / self.batchsize
            self.bo_grads += np.mean(delta[:, t, :], axis=0)
            delta_h = np.dot(delta[:, t, :], self.wo.T)
            # Backpropagation through time
            for bptt_step in np.arange(t + 1)[::-1]:
                XH = np.concatenate((self.X[:, bptt_step, :], self.Hs[:, bptt_step-1, :]), axis=1)
                XHP = np.concatenate((self.X[:, bptt_step, :],
                                      self.states_r[:,bptt_step,:]*self.Hs[:, bptt_step - 1, :]), axis=1)
                dz = (- self.Hs[:,bptt_step-1,:] + self.states_g[:,bptt_step,:]) * delta_h
                dg = self.states_z[:, bptt_step, :] * delta_h
                dxr = np.dot(dg * Tanh().deriv(self.states_g[:,bptt_step,:]), self.gates_wg.T)
                dr = dxr[:, self.inshape[-1]:] * self.Hs[:,bptt_step-1,:]
                # diffs w.r.t. vector inside sigma / tanh function
                dz_input = Sigmoid().deriv(self.states_z[:,bptt_step,:]) * dz
                dr_input = Sigmoid().deriv(self.states_r[:,bptt_step,:]) * dr
                dg_input = Tanh().deriv(self.states_g[:,bptt_step,:]) * dg
                # diffs w.r.t. inputs
                self.gates_wz_grads += np.dot(XH.T, dz_input) / self.batchsize
                self.gates_wr_grads += np.dot(XH.T, dr_input) / self.batchsize
                self.gates_wg_grads += np.dot(XHP.T, dg_input) / self.batchsize
                self.gates_bz_grads += np.mean(dz_input)
                self.gates_br_grads += np.mean(dr_input)
                self.gates_bg_grads += np.mean(dg_input)
                # compute bottom diff
                dxh = np.zeros(XH.shape)
                dxh += np.dot(dz_input, self.gates_wz.T)
                dxh += np.dot(dr_input, self.gates_wr.T)
                # save bottom diffs
                dh = dxh[:,self.inshape[-1]:]
                dh += (1 - self.states_z[:, bptt_step, :]) * delta_h
                dh += dxr[:, self.inshape[-1]:] * self.states_r[:,bptt_step,:]
                delta_h = dh
                next_delta[:, bptt_step, :] += dxh[:,:self.inshape[-1]]
        return next_delta

    def applygrad(self, lr=1e-4, wd=4e-4):
        self.gates_wz *= (1. - wd)
        self.gates_wr *= (1. - wd)
        self.gates_wg *= (1. - wd)
        self.gates_bz *= (1. - wd)
        self.gates_br *= (1. - wd)
        self.gates_bg *= (1. - wd)
        self.wo *= (1. - wd)
        self.bo *= (1. - wd)
        self.gates_wz -= lr * self.gates_wz_grads
        self.gates_wr -= lr * self.gates_wr_grads
        self.gates_wg -= lr * self.gates_wg_grads
        self.gates_bz -= lr * self.gates_bz_grads
        self.gates_br -= lr * self.gates_br_grads
        self.gates_bg -= lr * self.gates_bg_grads
        self.wo -= lr * self.wo_grads
        self.bo -= lr * self.bo_grads
        self.gates_wz_grads = np.zeros(self.gates_wz.shape)
        self.gates_wr_grads = np.zeros(self.gates_wr.shape)
        self.gates_wg_grads = np.zeros(self.gates_wg.shape)
        self.gates_bz_grads = np.zeros(self.gates_bz.shape)
        self.gates_br_grads = np.zeros(self.gates_br.shape)
        self.gates_bg_grads = np.zeros(self.gates_bg.shape)
        self.wo_grads = np.zeros(self.wo.shape)
        self.bo_grads = np.zeros(self.bo.shape)


if __name__ == "__main__":
    pass