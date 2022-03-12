
import numpy as np
import scipy as sp
from utils import *
from optimizer import *


class LinearModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y=None, args=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X, args=None):
        raise NotImplementedError()


class PCA(LinearModel):
    """ Principal component analysis """

    def __init__(self):
        self.PC = None
        self.SV = None

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        mX = np.mean(X, axis=0, keepdims=True)
        X0 = X - np.repeat(mX, N, axis=0)
        if N >= P:
            covX = np.dot(X0.T, X0)/(N-1)
            PC, SV, V = sp.linalg.svd(covX)
        else:
            _, D, PC = sp.linalg.svd(X0)
            PC = PC.T
            M = min(N, P)
            SV = np.zeros(P)
            SV[0:M] = D[0:M]
        self.PC = PC
        self.SV = SV
        return PC, SV

    def predict(self, X, args=None):
        return np.dot(X, self.PC)


class kNN(LinearModel):
    """k-nearest neighbor classifier"""

    def __init__(self, num_classes=10, k=10):
        super().__init__()
        self.k = k
        self.num_classes = num_classes

    def fit(self, X, y=None, args=None):
        self.X = X
        self.y = y

    def predict(self, X, args=None):
        return super().predict(X, args)


class BernoulliNB(LinearModel):
    """Naive Bayesian classifier when feature vectors are binary
    X: N by P feature matrix, N number of samples, P number of features
    y: N by 1 target vector
    Refer to P. 69 of PRML
    """

    def __init__(self, num_classes=10, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.priors = np.zeros(num_classes, dtype=float) # K by 1

    def fit(self, X, y=None, args=None):
        self.thetas = np.zeros((self.num_classes, X.shape[-1]), dtype=float) # K by P
        self.class_counts = np.zeros(self.num_classes, dtype=int) # K by 1
        for k in range(self.num_classes):
            idx = np.argwhere(y==k)
            self.class_counts[k] = len(idx)
            self.thetas[k, :] = np.mean(X[idx, :], axis=0)
        # Laplace smoothing
        self.priors = (self.class_counts+1.) / (np.sum(self.class_counts+1.))
        self.log_priors = np.log(self.priors + self.eps) # K by 1

    def predict(self, X, args=None):
        X0 = not X
        logT1 = np.log(self.thetas + self.eps) # K by P
        logT0 = np.log(1 - self.thetas + self.eps) # K by P
        yp = np.zeros((X.shape[0], self.num_classes), dtype=float) # N by K
        for k in range(self.num_classes):
            L1 = np.zeros_like(X) # N by P
            L0 = np.zeros_like(X0) # N by P
            for n in range(X.shape[0]):
                L1[n, :] = X[n, :] * logT1[k, :]
                L0[n, :] = X0[n, :] * logT0[k, :]
            yp[:, k] = np.sum(L1+L0, axis=-1) + self.log_priors[k] # N by 1
        yy = np.argmax(softmax(yp), axis=-1)
        return yy, yp


class GaussianNB(LinearModel):
    """Naive Bayesian classifier when feature vectors are continuous
    X: N by P feature matrix, N number of samples, P number of features
    y: N by 1 target vector
    """

    def __init__(self, num_classes=10, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.priors = np.zeros(num_classes, dtype=float) # K by 1

    def fit(self, X, y=None, args=None):
        self.mu = np.zeros((self.num_classes, X.shape[-1]), dtype=float) # K by P
        self.sigma2 = np.zeros((self.num_classes, X.shape[-1]), dtype=float) # K by P
        self.class_counts = np.zeros(self.num_classes, dtype=int) # K by 1
        for k in range(self.num_classes):
            idx = np.argwhere(y==k).squeeze()
            self.class_counts[k] = len(idx)
            self.mu[k, :] = np.mean(X[idx, :], axis=0)
            self.sigma2[k, :] = np.var(X[idx, :], axis=0)
        self.sigma2 += self.sigma2.max() * self.eps # variance smoothing
        # Laplace smoothing
        self.priors = (self.class_counts+1.) / (np.sum(self.class_counts+1.))
        self.log_priors = np.log(self.priors + self.eps)

    def predict(self, X, args=None):
        yp = np.zeros((X.shape[0], self.num_classes), dtype=float) # N by K
        for k in range(self.num_classes):
            X1 = X - np.repeat(self.mu[k:k+1, :], repeats=X.shape[0], axis=0)
            inv_Sigma = np.diag(1. / self.sigma2[k, :])
            yp[:, k] = -0.5 * np.diag(np.dot(np.dot(X1, inv_Sigma), np.transpose(X1))) \
                       -0.5 * np.sum(np.log(self.sigma2[k, :])) + self.log_priors[k] # N by 1
        yy = np.argmax(softmax(yp), axis=-1)
        return yy, yp


class RidgeRegression(LinearModel):
    """ Ridge regression
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 target vector
    # b: P by 1 regression coefficients
    # b0: the intercept
    """

    def __init__(self, wd=1e-4):
        self.wd = wd
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        PHI = np.concatenate((np.ones([N,1]), X), axis=1)
        if (P > N):
            invC = woodburyinv(self.wd * np.eye(P+1), PHI.T, PHI, np.eye(N))
        else:
            invC = np.linalg.inv(self.wd * np.eye(P+1) + np.dot(PHI.T, PHI))
        w = np.dot(np.dot(invC, PHI.T), y)
        self.w, self.b = w[1:], w[0]
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return yp


class LDAClassifier(LinearModel):
    """ Fisher's Linear Discriminant Analysis
    # y: N by 1 labels in {-1, 1}
    # X: N by P matrix, N observation of P dimensional feature vectors
    # wd: weight decay coefficient
    """

    def __init__(self, wd=1e-4):
        self.wd = wd
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        X1 = X[np.argwhere(y == 1), :]
        X2 = X[np.argwhere(y == -1), :]
        mu1 = np.squeeze(np.mean(X1, axis=0))
        mu2 = np.squeeze(np.mean(X2, axis=0))
        Sw = np.cov(np.transpose(X))
        self.w = np.dot(np.linalg.inv(Sw + self.wd * np.eye(P)), (mu1 - mu2).T)
        self.b = -np.dot(mu1 + mu2, self.w) / 2
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return np.sign(yp), yp


class LogisticRegression(LinearModel):
    """ Logistic regression for binary classification (Page 205-208 of PRML)
    # X: N by P design matrix with N samples of M features
    # y: N by 1 labels in {-1, 1}
    # wd: weight decay coefficient
    # w: P by 1 weight vector
    # b: bias
    """

    def __init__(self, wd=1e-4, optimizer=LbfgsOptimizer(), verbose=False):
        self.wd = wd
        self.optimizer = optimizer
        self.verbose = verbose
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        # add a constant column to cope with bias
        PHI = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
        N, P = PHI.shape
        # initialization
        w = np.zeros(P)  # rough initialization

        if isinstance(self.optimizer, GradientDescentOptimizer):
            # Logistic regression solved by gradient-based optimizer
            # min sum(log(1 + exp(-t.*(PHI * W)))) + wd *norm(w)
            w = self.optimizer.minimize(self._logistic_cost,
                                        w, args=(self.wd, PHI, y))
        elif isinstance(self.optimizer, str) and \
                self.optimizer.lower() == 'irls':
            # Iterative reweighted least square (IRLS) by Newton-Raphson
            # iterative optimization scheme.
            # w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
            y[np.argwhere(y == -1)] = 0  # here class label should be in {0, 1}
            w[0] = np.log(np.mean(y) / (1 - np.mean(y)))

            # stop conditions
            d_w = np.Inf
            maxit = 500
            stopeps = 1e-6

            i = 1
            while (d_w > stopeps) and (i < maxit):
                wold = w
                # predicted target value
                t = 1 / (1 + np.exp(-np.dot(PHI, w)))
                # the variance matrix of target value
                R = np.diag(np.squeeze(t * (1 - t)))
                # update with a norm2 regularization of w
                # H = PHI'*R*PHI + wd*eye(P);
                if (P > N):
                    invH = woodburyinv(self.wd * np.eye(P), PHI.T, PHI, R)
                else:
                    invH = np.linalg.inv(self.wd * np.eye(P) +
                                         np.dot(np.dot(PHI.T, R), PHI))

                w = w - np.dot(invH, np.dot(PHI.T, t - y) + self.wd * w)
                d_w = np.linalg.norm(wold - w)

                if self.verbose:
                    print('Iteration %d: wchange = %f' % (i, d_w))
                i = i + 1

            if (i >= maxit):
                print('Optimization end with maximum iterations = %d' % maxit)
        else:
            assert False, 'Unknown optimizer'

        self.w, self.b = w[1:], w[0]
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = sigmoid(np.dot(X, self.w) + self.b) - 0.5
        return np.sign(yp), yp

    def _logistic_cost(self, w, *args):
        wd, PHI, y = args
        y = np.squeeze(y)
        z = y * np.matmul(PHI, w)
        t = 1 / (1 + np.exp(-z))
        grad = np.matmul(-PHI.T, y * (1 - t)) + wd * w
        cost = -np.sum(np.log(t)) + 0.5 * wd * np.dot(w.T, w)
        return cost, grad


class SoftmaxClassifier(LinearModel):
    """ Softmax regression using stocastic gradient descent algorithm
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 class labels (t=k indicate belong to class k)
    # wd: weight decay coefficient
    # W: P by K regression coefficients
    """

    def __init__(self, wd=1e-4, optimizer=LbfgsOptimizer(), verbose=False):
        self.wd = wd
        self.optimizer = optimizer
        self.verbose = verbose
        self.W, self.b = None, 0

    def fit(self, X, y=None, args=None):
        K = len(np.unique(y))
        if len(y.shape) == 1 or y.shape[1] == 1:
            y = onehot(y, K)
        # add a constant column to cope with bias
        PHI = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
        N, P = PHI.shape
        W = np.ones([P, K])  # rough initialization
        theta = W.flatten()

        # cost, grad = _softmaxCost(theta, wd, PHI, y)
        # grad1 = numgrad(_softmaxCost, theta, wd, PHI, y)
        # diff = np.linalg.norm(grad1 - grad) / np.linalg.norm(grad1 + grad)
        # print(diff)

        opttheta = self.optimizer.minimize(self._softmax_cost,
                                           theta, args=(self.wd, PHI, y))
        W = np.reshape(opttheta, W.shape)
        self.W, self.b = W[1:,:], W[0, :]
        return self.W, self.b

    def predict(self, X, args=None):
        assert self.W is not None, 'Please fit the model before use'
        t = softmax(np.matmul(X, self.W) + self.b)
        y = np.argmax(t, axis=1)
        return y, t

    def _softmax_cost(self, theta, *args):
        """ Cross entropy error cost function """
        epsilon = 1e-32
        wd, PHI, y = args
        N, P = PHI.shape
        W = np.reshape(theta, [P, -1])
        t = softmax(np.matmul(PHI, W)) + epsilon
        grad = (1. / N) * np.matmul(PHI.T, t - y) + wd * W
        grad = grad.flatten()
        cost = -(1. / N) * np.dot(y.flatten().T, np.log(t.flatten())) \
               + 0.5 * wd * np.sum(W.flatten() ** 2)
        return cost, grad


if __name__ == "__main__":
    
    #Loading the Dataset
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as GNB
    from sklearn import metrics

    dataloader = load_breast_cancer()
    # keeping 80% as training data and 20% as testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        dataloader.data, dataloader.target, test_size=0.2, random_state=20)

    model0 = GNB()
    model = GaussianNB(num_classes=2)
    # model = SoftmaxClassifier()

    model0.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred0 = model0.predict(X_test)
    y_pred, _ = model.predict(X_test)

    accu0 = metrics.accuracy_score(y_pred0, y_test)
    accu = metrics.accuracy_score(y_pred, y_test)
    # accu = np.sum(np.equal(y_pred, y_test))/len(y_test)
    print(f"accu0 is {accu0} and accu is {accu}")
