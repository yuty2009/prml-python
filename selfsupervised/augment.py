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


def get_transforms(type='', size=224, mean_std=None, color_jitter_s=1.0):
    s = color_jitter_s
    normalize = None
    if mean_std is not None:
        normalize = transforms.Normalize(
            mean=mean_std[0], std=mean_std[1])

    if str.lower(type) in ['', 'test', 'eval', 'val']:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif str.lower(type) in ['train', 'training']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    elif str.lower(type) in ['simclr', 'simclr_v1', 'moco', 'moco_v1']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([
                    transforms.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s),
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize
            ])
    elif str.lower(type) in ['mocov2', 'moco_v2', 'byol', 'simsiam']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(), # with 0.5 probability
                transforms.RandomApply([
                    transforms.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # the image size of cifar10 is too small to apply gaussian blur
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                normalize
            ])
        

def get_multicrop_transforms(size_crops, num_crops, mean_std=None, color_jitter_s=1.0):
    assert len(size_crops) == len(num_crops)

    s = color_jitter_s
    normalize = None
    if mean_std is not None:
        normalize = transforms.Normalize(
            mean=mean_std[0], std=mean_std[1])

    trans = []
    for i in range(len(size_crops)):
        randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                # scale=(min_scale_crops[i], max_scale_crops[i]),
            )
        trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize])
            ] * num_crops[i])
    
    return trans
