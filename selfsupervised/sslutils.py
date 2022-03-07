# Refer to https://github.com/leftthomas/SimCLR/blob/master/linear.py
import os
import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets

import common.distributed as dist
import common.torchutils as utils
import lars
import augment
import moco, simclr, byol, simsiam


class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes, n_features=2048, pretrained=None, freeze_feature=True):
        super(LinearClassifier, self).__init__()
        # encoder
        self.encoder = encoder
        if freeze_feature: # freeze all layers but the last fc
            for param in self.encoder.parameters():
                param.requires_grad = False
        # classifier
        self.fc = nn.Linear(n_features, num_classes)
        if pretrained and os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = utils.convert_state_dict(checkpoint['state_dict'])
            msg = self.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    def forward(self, x):
        x = self.encoder(x)
        out = self.fc(x)
        return out


def get_train_dataset(args):
    if str.lower(args.dataset) in ['cifar10', 'cifar-10']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size, args.mean_std)),
        )

    elif str.lower(args.dataset) in ['cifar100', 'cifar-100']:
        args.image_size = 32
        args.mean_std = ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023))
        train_dataset = datasets.CIFAR100(
            args.data_dir, train=True, download=True,
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size, args.mean_std)),
        )

    elif str.lower(args.dataset) in ['stl10', 'stl-10']:
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.data_dir, split="unlabeled", download=True,
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size, args.mean_std)),
        )

    elif str.lower(args.dataset) in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size, args.mean_std)),
        )

    else:
        raise NotImplementedError

    return train_dataset


def get_base_encoder(base_encoder, args):
    module_list = []
    for name, module in base_encoder.named_children():
        if str.lower(args.dataset) not in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
        if isinstance(module, nn.Linear):
            args.encoder_dim = module.weight.shape[1]
            continue
        module_list.append(module)
    module_list.append(nn.Flatten(1))
    base_encoder = nn.Sequential(*module_list)
    return base_encoder
    

def get_ssl_model_and_criterion(base_encoder, args):
    if str.lower(args.ssl) in ['moco', 'mocov1', 'moco_v1']:
        args.dim = 128
        args.moco_k = 4096 # 4096 for cifar10
        args.moco_m = 0.99 # 0.99 for cifar10
        args.moco_t = 0.1 # 0.1 for cifar10
        args.mlp = False
        args.schedule = 'step'
        model = moco.MoCo(
            base_encoder, args.encoder_dim, args.dim,
            args.moco_k, args.moco_m, args.moco_t, args.mlp,
            args.symmetric)
        criterion = nn.CrossEntropyLoss()

    elif str.lower(args.ssl) in ['mocov2', 'moco_v2']:
        args.dim = 128
        args.moco_k = 4096 # 4096 for cifar10, 65536 for ImageNet
        args.moco_m = 0.99 # 0.99 for cifar10
        args.moco_t = 0.07
        args.mlp = True
        args.schedule = 'cos'
        model = moco.MoCo(
            base_encoder, args.encoder_dim, args.dim,
            args.moco_k, args.moco_m, args.moco_t, args.mlp,
            args.symmetric)
        criterion = nn.CrossEntropyLoss()

    elif str.lower(args.ssl) in ['simclr', 'simclr_v1']:
        args.dim = 128
        model = simclr.SimCLR(
            base_encoder, args.encoder_dim, args.dim)
        criterion = nn.CrossEntropyLoss()

    elif str.lower(args.ssl) in ['byol']:
        args.dim = 256
        args.byol_m = 0.9
        model = byol.BYOL(
            base_encoder, args.encoder_dim, args.dim,
            m=args.byol_m)
        criterion = nn.CosineSimilarity(dim=1)
    
    elif str.lower(args.ssl) in ['simsiam']:
        args.dim = 256
        model = simsiam.SimSiam(
            base_encoder, args.encoder_dim, args.dim)
        criterion = nn.CosineSimilarity(dim=1)
    
    else:
        raise NotImplementedError

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    return model, criterion


def train_epoch_ssl(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss, total_num = 0.0, 0

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    train_bar = tqdm.tqdm(train_loader) if show_bar else train_loader

    for images, _ in train_bar:

        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        if str.lower(args.ssl) in ['byol', 'simsiam']: # without negative samples
            p1, p2, t1, t2 = model(images[0], images[1])
            loss = -0.5 * (criterion(p1, t2) + criterion(p2, t1))
        else: # 'moco'
            if hasattr(args, 'symmetric') and args.symmetric:
                p1, p2, t1, t2 = model(images[0], images[1])
                loss = -0.5 * (criterion(p1, t1) + criterion(p2, t2))
            else:
                p1, t1 = model(images[0], images[1])
                loss = criterion(p1, t1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = dist.all_reduce(loss)
        total_loss += loss.item()
        total_num += images[0].size(0)

        if show_bar:
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.4f}".format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


def get_optimizer(model, args):
    """  """
    if str.lower(args.optimizer) == "lars": 
        optimizer = lars.LARS(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            momentum=0.9, max_epoch=args.epochs,
            warmup_epochs=round(0.1*args.epochs))
    elif str.lower(args.optimizer) == "sgd":
         optimizer = torch.optim.SGD(
             model.parameters(), lr=args.lr,
             weight_decay=args.weight_decay, momentum=0.9)
    elif str.lower(args.optimizer) == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    else: 
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
