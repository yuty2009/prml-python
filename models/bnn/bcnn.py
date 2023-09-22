# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/

import numpy as np
from deeplearning.bnn.bayeslayers import *


class BayesLeNet5(nn.Module):
    """Convolutional Neural Network with Bayes By Backprop"""

    def __init__(self, num_classes=10):
        super(BayesLeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes)
            # BayesLinear(84, num_classes, prior=GaussPrior(0, 0.5))
        )

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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x, loss_1 = self.classifier(x), 0
        return x, loss_1

    def predict_mcmc(self, X, n_samples):

        predictions = X.data.new(n_samples, X.shape[0], self.output_dim)
        loss_kl = np.zeros(n_samples)

        for i in range(n_samples):
            y, loss_kl_1 = self.forward(X, sample=True)
            predictions[i] = y
            loss_kl[i] = loss_kl_1

        return torch.mean(predictions, dim=0), loss_kl


if __name__ == "__main__":

    import os
    import datetime
    import argparse
    from torchvision import datasets, transforms
    from torch.utils.tensorboard import SummaryWriter
    import common.torchutils as utils
    from engine import train_epoch, evaluate

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--dataset-dir', default='e:/prmldata/mnist', type=str, help='dataset directory')
    parser.add_argument('--arch', default='bayeslenet5', type=str, help='model architecture')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-8, type=float, help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs to warmup LR')
    parser.add_argument('--schedule', default='cos', type=str, help='learning rate schedule (how to change lr)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--save-freq', default=50, type=int, help='save frequency')

    args = parser.parse_args()

    if not hasattr(args, 'output_dir'):
        args.output_dir = os.path.join(args.dataset_dir, 'output')

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.dataset_dir))

    tf_train = transforms.Compose([transforms.ToTensor()])
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(args.dataset_dir, train=True, download=True, transform=tf_train)
    test_dataset = datasets.MNIST(args.dataset_dir, train=False, download=True, transform=tf_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BayesLeNet5(num_classes=10).to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log/{args.arch}"))

    print("=> begin training")
    args.best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss, train_accu1 = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_loss, test_accu1 = evaluate(test_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = test_accu1 > args.best_acc
        args.best_acc = max(test_accu1, args.best_acc)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, epoch + 1,
                is_best=is_best,
                save_dir=os.path.join(args.output_dir, f"checkpoint/{args.arch}"))

        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Loss/test", test_loss, epoch)
            args.writer.add_scalar("Accu/train", train_accu1, epoch)
            args.writer.add_scalar("Accu/test", test_accu1, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)
