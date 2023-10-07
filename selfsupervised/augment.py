# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from PIL import Image, ImageFilter, ImageOps

import torch
import torchvision.transforms as transforms


class TransformContrast:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TransformMultiCrops:
    """Take multiple random crops of one image."""

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops
    

class TransformMultiViews(object):
    def __init__(self, base_transform, num_views=4):
        self.num_views = num_views
        self.base_transform = base_transform
      
    def __call__(self, x):
        x_views = [self.base_transform(x) for _ in range(self.num_views)]
        return x_views


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class Solarization(object):
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


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
    elif str.lower(type) in ['ssl']:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.2*s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # the image size of cifar10 is too small to apply gaussian blur
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                transforms.RandomApply([Solarization()], p=0.1),
                transforms.ToTensor(),
                normalize
            ])
        

def get_multicrop_transforms(
    size_crops, num_crops, min_scales=None, max_scales=None,
    mean_std=None, color_jitter_s=1.0):

    if min_scales is None: min_scales = [0.08] * len(num_crops)
    if max_scales is None: max_scales = [1.00] * len(num_crops)

    assert len(size_crops) == len(num_crops)
    assert len(min_scales) == len(num_crops)
    assert len(max_scales) == len(num_crops)

    s = color_jitter_s
    normalize = None
    if mean_std is not None:
        normalize = transforms.Normalize(
            mean=mean_std[0], std=mean_std[1])

    trans = []
    for i in range(len(size_crops)):
        randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scales[i], max_scales[i]),
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


class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        transforms,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__()
        self.return_index = return_index
        self.trans = transforms
        self.dataset = dataset

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def __len__(self):
        return len(self.dataset)
