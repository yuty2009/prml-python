# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torchvision.transforms as transforms


class TransformContrast:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


CACHED_MEAN_STD = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}


def get_transforms(type='', size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if str.lower(type) in ['', 'test', 'eval', 'val']:
        return transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif str.lower(type) in ['train', 'training']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif str.lower(type) in ['simclr', 'simclr_v1']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
    elif str.lower(type) in ['moco', 'mocov1', 'moco_v1']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    elif str.lower(type) in ['mocov2', 'moco_v2', 'byol', 'simsiam']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
        