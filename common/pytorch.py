
import copy
import tqdm
import torch
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    """ Softmax regression using stocastic gradient descent algorithm
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 class labels (t=k indicate belong to class k)
    # num_classes: number of classes
    # feature_dim: feature dimension
    # epochs: epochs to train
    # lr: learning rate
    # wd: weight decay coefficient
    """

    def __init__(self, num_classes, feature_dim, epochs=200, lr=0.08, wd=5e-4, verbose=False):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.verbose = verbose
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.model = nn.Linear(feature_dim, num_classes)

    def fit(self, X, y=None):
        assert X.size(1) == self.feature_dim, 'input dimension mismatch'
        if y is not None:
            assert X.size(0) == len(y), 'length of X and y must be the same'
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.wd)

        for epoch in range(self.epochs):
            yp = self.model(X)
            loss = torch.nn.functional.cross_entropy(yp, y, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.verbose:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss {loss.item():.4f}")

    def forward(self, X):
        with torch.no_grad():
            t = self.model(X)
            y = t.argmax(dim=1)
        return y, t


class KNNClassifier(nn.Module):
    def __init__(self, num_classes, knn_k=200, knn_t=0.1):
        super(KNNClassifier, self).__init__()
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.num_classes = num_classes
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def forward(self, x):
        assert self.x_train is not None and self.y_train is not None, "call fit first"
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(x, self.x_train.t().contiguous())
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(self.y_train.expand(x.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.knn_t).exp()
        # counts for each class
        one_hot_label = torch.zeros(x.size(0) * self.knn_k, self.num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
    

class KMeansCluster(nn.Module):
    def __init__(self, num_centroids, epochs=20, verbose=False):
        super(KMeansCluster, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.num_centroids = num_centroids
    
    def forward(self, x):
        num_examples = x.shape[0]
        assert num_examples >= self.epochs, "number of samples should be larger than k"
        centroids = copy.deepcopy(x[:self.num_centroids])
        indices = torch.zeros(num_examples, dtype=int, device=x.device)
        for it in range(self.epochs):
            indices_old = copy.deepcopy(indices)
            dist = torch.mm(x, centroids.T)
            index_list = [[] for i in range(self.num_centroids)]
            progress_bar = tqdm.tqdm(range(num_examples))
            progress_bar.set_description("Clustering [{}/{}]".format(it+1, self.epochs))
            # update cluster assignments
            for i in progress_bar:
                ci = torch.argmin(dist[i])
                indices[i] = ci
                index_list[ci].append(i)
            # update centroids
            for j in range(self.num_centroids):
                if len(index_list[j]) > 0:
                    centroids[j] = torch.mean(x[index_list[j],:], dim=0, keepdim=True)
            # stop iteration if assignments do not change
            if torch.equal(indices, indices_old):
                if self.verbose:
                    print(f"clustering ended after {it+1} iterations")
                break
        if it >= self.epochs and self.verbose:
            print(f"clustering ended due to maximum iterations")
        return indices
    