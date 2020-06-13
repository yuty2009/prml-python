# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/

from deeplearning.bnn.bayeslayers import *


class BayesLeNet5(nn.Module):
    """Convolutional Neural Network with Bayes By Backprop"""

    def __init__(self, num_classes=10):
        super(BayesLeNet5, self).__init__()
        self.output_dim = num_classes

        self.conv1 = BayesConv2d(1, 32, kernel_size=(3, 3), prior=GaussPrior(0, 0.5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = BayesConv2d(32, 64, kernel_size=(3, 3), prior=GaussPrior(0, 1.0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.drop1 = nn.Dropout(0.25)
        self.dense1 = BayesLinear(64 * 5 * 5, 128, prior=GaussPrior(0, 0.5))
        self.dense2 = BayesLinear(128, 84, prior=GaussPrior(0, 0.5))
        # self.drop2 = nn.Dropout(0.5)
        self.dense3 = BayesLinear(84, num_classes, prior=GaussPrior(0, 0.5))

    def forward(self, x, sample=False):
        loss_kl = 0
        x, loss_1 = self.conv1(x, sample)
        loss_kl += loss_1
        x = self.pool1(F.relu(x))
        x, loss_1 = self.conv2(x, sample)
        loss_kl += loss_1
        x = self.pool2(F.relu(x))
        x = x.view(x.size(0), -1)  # flatten the tensor
        # x = self.drop1(x)
        x, loss_1 = self.dense1(x, sample)
        loss_kl += loss_1
        x = F.relu(x)
        x, loss_1 = self.dense2(x, sample)
        loss_kl += loss_1
        x = F.relu(x)
        # x = self.drop2(x)
        x, loss_1 = self.dense3(x, sample)
        loss_kl += loss_1
        return x, loss_kl

    def predict_mcmc(self, X, n_samples):

        predictions = X.data.new(n_samples, X.shape[0], self.output_dim)
        loss_kl = np.zeros(n_samples)

        for i in range(n_samples):
            y, loss_kl_1 = self.forward(X, sample=True)
            predictions[i] = y
            loss_kl[i] = loss_kl_1

        return torch.mean(predictions, dim=0), loss_kl


if __name__ == "__main__":

    import argparse
    import torch.optim as optim
    from utils.mnistreader import *

    f_train_images = 'e:/prmldata/mnist/train-images-idx3-ubyte'
    f_train_labels = 'e:/prmldata/mnist/train-labels-idx1-ubyte'
    f_test_images = 'e:/prmldata/mnist/t10k-images-idx3-ubyte'
    f_test_labels = 'e:/prmldata/mnist/t10k-labels-idx1-ubyte'

    imsize = 28
    mnist = MNISTReader(f_train_images, f_train_labels, f_test_images, f_test_labels)
    trainset = mnist.get_train_dataset(onehot_label=False,
                                       reshape=True, new_shape=(-1, imsize, imsize, 1),
                                       tranpose=True, new_pos=(0, 3, 1, 2))
    testset = mnist.get_test_dataset(onehot_label=False,
                                     reshape=True, new_shape=(-1, imsize, imsize, 1),
                                     tranpose=True, new_pos=(0, 3, 1, 2))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser()
    help_ = "Load model checkpoints"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()

    cnn = BayesLeNet5(num_classes=10).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3)

    if args.weights:
        print('=> loading checkpoint %s' % args.weights)
        checkpoint = torch.load(args.weights)
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint %s' % args.weights)
    else:
        # train
        verbose = 0
        epochs = 100
        batch_size = 100
        num_batches = np.ceil(trainset.num_examples / batch_size).astype('int')
        weight_kl = 1.0 / num_batches
        loss_test_list = [np.Inf]
        for epoch in range(epochs):
            loss_train = 0
            loss_kl_train = 0
            correct_train = 0
            for step in range(num_batches):
                X_batch, y_batch = trainset.next_batch(batch_size)
                X_batch = torch.tensor(X_batch, device=device)
                y_batch = torch.tensor(y_batch, device=device)

                yp_batch, loss_kl = cnn(X_batch, sample=True)
                loss_ce = F.cross_entropy(yp_batch, y_batch.long(), reduction='sum')
                loss = loss_ce + weight_kl * loss_kl
                loss_train += loss_ce
                loss_kl_train += loss_kl

                yp_batch = yp_batch.argmax(dim=1, keepdim=True)
                correct_train += yp_batch.eq(y_batch.view_as(yp_batch)).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics every 100 steps
                if verbose == 1 and (step + 1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], CrossEntropy Loss: {:.4f}, "
                          "KL Div: {:.4f} Total loss {:.4f}"
                          .format(epoch + 1, epochs,
                                  (step + 1) * batch_size, trainset.num_examples,
                                  loss_ce.item(), loss_kl.item(), loss.item()))

            acc_train = correct_train / trainset.num_examples

            with torch.no_grad():
                loss_test = 0
                correct_test = 0
                testset.reset()
                while testset.epochs_completed <= 0:
                    X_batch, y_batch = testset.next_batch(1000)
                    X_batch = torch.tensor(X_batch, device=device)
                    y_batch = torch.tensor(y_batch, device=device)
                    yp_batch, _ = cnn(X_batch)
                    # yp_batch, _ = cnn.predict_mcmc(X_batch, n_samples=100)
                    loss_ce = F.cross_entropy(yp_batch, y_batch.long(), reduction='sum')
                    loss_test += loss_ce
                    # get the index of the max log-probability
                    yp_batch = yp_batch.argmax(dim=1, keepdim=True)
                    correct_test += yp_batch.eq(y_batch.view_as(yp_batch)).sum().item()

                acc_test = correct_test / testset.num_examples

                print("Epoch [{}/{}], Training Loss: {:.4f}, "
                      "KL Div: {:.4f}, Training Accuracy {:.4f}, "
                      "Validation Loss: {:.4f}, Validation Accuracy {:.4f}"
                      .format(epoch + 1, epochs,
                              loss_train.item(), loss_kl_train.item(), acc_train,
                              loss_test.item(), acc_test))

                if loss_test.item() < min(loss_test_list):
                    print("Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ..."
                          .format(min(loss_test_list), loss_test.item()))
                    torch.save({'state_dict': cnn.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               'bcnn_mnist_ckpt')
                loss_test_list.append(loss_test.item())


