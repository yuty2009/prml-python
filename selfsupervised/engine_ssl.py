
import os
import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets

import common.distributed as dist
import common.torchutils as utils
import lars
import augment
import ntxent
import moco, simclr, swav, byol, simsiam, dino


# implementation follows https://github.com/leftthomas/SimCLR/blob/master/linear.py
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes, n_features=2048, pretrained=None, freeze_feature=True):
        super(LinearClassifier, self).__init__()
        # encoder
        self.encoder = encoder
        if freeze_feature: # freeze all layers but the last fc
            for param in self.encoder.parameters():
                param.requires_grad = False
        # load pretrained model
        self.load_pretrained(pretrained)
        # classifier
        self.fc = nn.Linear(n_features, num_classes)
        torch.nn.init.trunc_normal_(self.fc.weight, std=2e-5)

    def load_pretrained(self, pretrained):
        if pretrained and os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = utils.convert_state_dict(checkpoint['state_dict'])
            msg = self.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.normalize(x, dim=1)
        out = self.fc(x)
        return out


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch
#                    and https://github.com/leftthomas/SimCLR
#                    and https://colab.research.google.com/github/facebookresearch/
#                        moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
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


def train_epoch_ssl(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss = 0.0

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    for step, (images, _) in enumerate(data_bar):

        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        if str.lower(args.ssl) in ['moco', 'mocov1', 'moco_v1', 'mocov2', 'moco_v2']:
            if hasattr(args, 'symmetric') and args.symmetric:
                p1, p2, t1, t2, queue = model(images[0], images[1])
                loss = 0.5 * (criterion(p1, t1, queue) + criterion(p2, t2, queue))
            else:
                p1, t1, queue = model(images[0], images[1])
                loss = criterion(p1, t1, queue)

        elif str.lower(args.ssl) in ['simclr', 'simclr_v1']: # big batch_size required
            p1, t1 = model(images[0], images[1])
            loss = 0.5 * (criterion(p1, t1) + criterion(t1, p1))

        elif str.lower(args.ssl) in ['swav', 'dino']: # multi-crop supported
            p1, t1 = model(images)
            loss = criterion(p1, t1)

        elif str.lower(args.ssl) in ['byol', 'simsiam']: # without negative samples
            p1, p2, t1, t2 = model(images[0], images[1])
            loss = -0.5 * (criterion(p1, t2).mean() + criterion(p2, t1).mean())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = dist.all_reduce(loss)
        total_loss += loss.item()

        if show_bar:
            data_bar.set_description(
                "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f}".format(
                    epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / (step+1)))

    return total_loss / len(data_loader)


def evaluate_ssl(memory_loader, test_loader, model, epoch, args):
    model.eval()

    num_classes = len(memory_loader.dataset.classes)
    knn = KNNClassifier(num_classes, args.knn_k, args.knn_t)

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        memory_bar = tqdm.tqdm(memory_loader, desc='Feature extracting') if show_bar else memory_loader
        for data, target in memory_bar:
            feature = model(data.to(args.device))
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank)
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank.device)
        knn.fit(feature_bank, feature_labels)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm.tqdm(test_loader) if show_bar else test_loader
        for data, target in test_bar:
            data, target = data.to(args.device), target.to(args.device)

            feature = model(data)
            feature = torch.nn.functional.normalize(feature, dim=1)
            pred_labels = knn(feature)

            total_num += data.size(0)
            total_top1 += torch.sum((pred_labels[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            if show_bar:
                test_bar.set_description(
                    "Test  Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                        epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


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
            module_list.append(nn.Flatten(1))
            continue
        module_list.append(module)
    base_encoder = nn.Sequential(*module_list)
    return base_encoder
    

def get_ssl_model_and_criterion(base_encoder, args):
    if str.lower(args.ssl) in ['moco', 'mocov1', 'moco_v1']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.n_mlplayers = 0
        args.hidden_dim = 128
        args.use_bn = False
        args.queue_size = 4096 # 4096 for cifar10, 65536 for ImageNet
        args.momentum = 0.9 # 0.9 for cifar10
        args.symmetric = True
        args.schedule = 'cos'
        args.temperature = 0.07 # 0.1 for cifar10
        model = moco.MoCo(
            encoder = base_encoder,
            encoder_dim = args.encoder_dim,
            feature_dim = args.feature_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
            queue_size = args.queue_size,
            momentum = args.momentum, 
            symetric = args.symmetric,
        )
        criterion = ntxent.NTXentLossWithQueue(args.temperature)

    elif str.lower(args.ssl) in ['mocov2', 'moco_v2']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.n_mlplayers = 2
        args.hidden_dim = 128
        args.use_bn = False
        args.queue_size = 4096 # 4096 for cifar10, 65536 for ImageNet
        args.momentum = 0.9 # 0.9 for cifar10
        args.symmetric = True
        args.schedule = 'cos'
        args.temperature = 0.1 # 0.1 for cifar10
        model = moco.MoCo(
            encoder = base_encoder,
            encoder_dim = args.encoder_dim,
            feature_dim = args.feature_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
            queue_size = args.queue_size,
            momentum = args.momentum, 
            symmetric = args.symmetric,
        )
        criterion = ntxent.NTXentLossWithQueue(args.temperature)

    elif str.lower(args.ssl) in ['simclr', 'simclr_v1']:
        args.lr = 1e-3 # 1e-3 for cifar10
        args.weight_decay = 1e-6 # 1e-6 for cifar10
        args.feature_dim = 512
        args.n_mlplayers = 2
        args.hidden_dim = 128
        args.use_bn = False
        args.temperature = 0.5
        model = simclr.SimCLR(
            encoder = base_encoder, 
            encoder_dim = args.encoder_dim, 
            feature_dim = args.feature_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
        )
        criterion = ntxent.NTXentLoss(args.temperature)

    elif str.lower(args.ssl) in ['swav']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.n_mlplayers = 2
        args.hidden_dim = 128
        args.use_bn = False
        args.temperature = 0.5
        model = swav.SwAV(
            encoder = base_encoder, 
            encoder_dim = args.encoder_dim, 
            feature_dim = args.feature_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
            n_prototypes = 30, # 30 for cifar10, 3000 for imagenet-1k
        )
        criterion = swav.SwAVLoss(2, args.temperature)

    elif str.lower(args.ssl) in ['byol']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.predict_dim = 512
        args.n_mlplayers = 2
        args.hidden_dim = 128
        args.use_bn = False
        args.momentum = 0.9
        model = byol.BYOL(
            encoder = base_encoder, 
            encoder_dim = args.encoder_dim, 
            feature_dim = args.feature_dim,
            predict_dim = args.predict_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
            momentum = args.momentum,
        )
        criterion = nn.CosineSimilarity(dim=1)
    
    elif str.lower(args.ssl) in ['simsiam']:
        args.lr = 6e-2 # 6e-2 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.predict_dim = 512
        args.n_mlplayers = 2
        args.hidden_dim = 128
        args.use_bn = False
        model = simsiam.SimSiam(
            encoder = base_encoder,
            encoder_dim = args.encoder_dim,
            feature_dim = args.feature_dim,
            predict_dim = args.predict_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            use_bn = args.use_bn,
        )
        criterion = nn.CosineSimilarity(dim=1)

    elif str.lower(args.ssl) in ['dino']:
        args.lr = 6e-2 # 1e-3 for cifar10
        args.weight_decay = 5e-4 # 5e-4 for cifar10
        args.feature_dim = 512
        args.n_mlplayers = 3
        args.hidden_dim = 128
        args.bottleneck_dim = 256
        args.use_bn = False
        args.momentum = 0.996
        model = dino.DINO(
            encoder = base_encoder, 
            encoder_dim = args.encoder_dim, 
            feature_dim = args.feature_dim,
            n_mlplayers = args.n_mlplayers,
            hidden_dim = args.hidden_dim,
            bottleneck_dim = args.bottleneck_dim,
            use_bn = args.use_bn,
            momentum = args.momentum,
        )
        criterion = dino.DINOLoss(out_dim=args.feature_dim)
    
    else:
        raise NotImplementedError

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    return model, criterion


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


def get_train_dataset(args, evaluate=False):
    if str.lower(args.dataset) in ['cifar10', 'cifar-10']:
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
            transform=augment.TransformContrast(
                augment.get_transforms('ssl', args.image_size, args.mean_std)),
        )
        if evaluate:
            memory_dataset = datasets.CIFAR10(
                args.data_dir, train=True, download=True,
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )
            test_dataset = datasets.CIFAR10(
                args.data_dir, train=False, download=True,
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )

    elif str.lower(args.dataset) in ['stl10', 'stl-10']:
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.data_dir, split="unlabeled", download=True,
            transform=augment.TransformContrast(
                augment.get_transforms('ssl', args.image_size, args.mean_std)),
        )
        if evaluate:
            memory_dataset = datasets.STL10(
                args.data_dir, split="train", download=True,
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )
            test_dataset = datasets.STL10(
                args.data_dir, split="test", download=True,
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )

    elif str.lower(args.dataset) in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.TransformContrast(
                augment.get_transforms('ssl', args.image_size, args.mean_std)),
        )
        if evaluate:
            memory_dataset = datasets.ImageFolder(
                os.path.join(args.data_dir, 'train'),
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )
            test_dataset = datasets.ImageFolder(
                os.path.join(args.data_dir, 'val'),
                transform=augment.get_transforms('test', args.image_size, args.mean_std)
            )

    else:
        raise NotImplementedError

    if evaluate:
        return train_dataset, memory_dataset, test_dataset
    else:
        return train_dataset
    