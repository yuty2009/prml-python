# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch.nn as nn
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


class Augmentation:
    @staticmethod
    def get(type='', size=224):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if type in ['', 'test']:
            return transforms.Compose(
                [
                    transforms.Resize(size=size),
                    transforms.ToTensor(),
                ]
            )

        elif type in ['simclr', 'SimCLR']:
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

        elif type in ['mocov1', 'moco_v1', 'MoCo', 'MoCov1', 'MoCo_v1']:
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

        elif type in ['mocov2', 'moco_v2', 'MoCov2', 'MoCo_v2']:
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
            