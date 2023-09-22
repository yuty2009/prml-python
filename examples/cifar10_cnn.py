
import os
import datetime
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import common.torchutils as utils
from models.cnn.cifarcnn import CIFARCNN


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--dataset-dir', default='e:/prmldata/cifar10', type=str, help='dataset directory')
parser.add_argument('--arch', default='cifarcnn', type=str, help='model architecture')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--min_lr', default=1e-8, type=float, help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str, help='learning rate schedule (how to change lr)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default=50, type=int, help='save frequency')


def main(args):
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.dataset_dir))

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    )

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    tf_test = transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=tf_train)
    test_dataset = datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=tf_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # create model
    print("=> creating model ")
    model = CIFARCNN(input_shape=(3, 32, 32), num_classes=10).to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if hasattr(args, 'resume') and args.resume:
        utils.load_checkpoint(args.resume, model, optimizer, args)
    else:
        print("=> going to train from scratch")

    if hasattr(args, 'evaluate') and args.evaluate:
        test_loss, test_accu1, test_accu5 = utils.evaluate(
            test_loader, model, criterion, 0, args)
        print(f"Test loss: {test_loss:.4f} Acc@1: {test_accu1:.2f} Acc@5 {test_accu5:.2f}")
        return

    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log/{args.arch}"))

    print("=> begin training")
    args.best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss, train_accu1 = utils.train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_loss, test_accu1 = utils.evaluate(test_loader, model, criterion, epoch, args)

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


if __name__ == '__main__':
    
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

    main(args)
