
import os
import glob
import torch
import numpy as np
from PIL import Image


label2index = {'Normal': 0, 'Benign': 0, 'InSitu': 1, 'Invasive': 1}

class BACHPatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='all', patch_size=124, transform=None, transform_patch=None):
        self.root = root
        self.test_ratio = 0.2
        self.patch_size = patch_size
        self.transform = transform
        self.transform_patch = transform_patch
        self.classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
        self.class_to_idx = label2index
        self.imgs = self.make_dataset(split)
        
    def make_dataset(self, split='all'):
        imgs = []
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(self.root, target)
            img_list = glob.glob(os.path.join(d, '*.tif'))
            if split != 'all':
                num_imgs = len(img_list)
                num_test = int(num_imgs * self.test_ratio)
                num_train = num_imgs - num_test
                img_train = img_list[:num_train]
                img_test = img_list[num_train:]
                if split == 'train':
                    img_list = img_train
                elif split == 'test':
                    img_list = img_test
            for path in img_list:
                imgs.append((os.path.join(d, path), self.class_to_idx[target]))
        return imgs
    
    def patchify(self, img, patch_size):
        """
        input img: (H, W, 3)
        output x: (num_patches, patch_size, patch_size, 3)
        """
        p = patch_size
        assert img.shape[0] % p == 0 and img.shape[1] % p == 0

        nh, nw = img.shape[0] // p, img.shape[1] // p
        x = img.reshape(nh, p, nw, p, 3)
        x = x.transpose(0, 2, 1, 3, 4)
        # x = torch.einsum('hpwqc->hwpqc', x)
        x = x.reshape(nh * nw, p, p, 3)
        return x

    def unpatchify(self, x, nh, nw):
        """
        input x: (num_patches, patch_size, patch_size, 3)
        output img: (H, W, 3)
        """
        assert nh * nw == x.shape[0] and x.shape[1] == x.shape[2]
        p = x.shape[1]
        
        x = x.reshape(nh, nw, p, p, 3)
        x = x.transpose(0, 2, 1, 3, 4)
        # x = torch.einsum('hwpqc->hpwqc', x)
        img = x.reshape(nh * p, nw * p, 3)
        return img
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        img_array = np.array(self.transform(img))
        patches = self.patchify(img_array, patch_size=self.patch_size)
        # patches = patches.transpose(0, 3, 1, 2) # ToTensor() will do this
        if self.transform is not None:
            patches_transformed = []
            for patch in patches:
                img_patch = Image.fromarray(patch)
                patch_transformed = self.transform_patch(img_patch)
                patches_transformed.append(patch_transformed)
            patches = torch.stack(patches_transformed)
        return patches, target
    
    def __len__(self):
        return len(self.imgs)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

if __name__ == '__main__':
    # Test the dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from PIL import Image
    
    root = 'f:/medicalimages/bach/ICIAR2018_BACH_Challenge/Photos'
    tf_resize = transforms.Resize((1488, 1984)) # transforms.CenterCrop((1488, 1984))
    tf_train = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = torchvision.datasets.ImageFolder(root, transform)
    dataset = BACHPatchDataset(root, split='test', patch_size=124, transform=tf_resize, transform_patch=tf_train)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        x1 = x[0].numpy().transpose(0, 2, 3, 1)
        x_grid = dataset.unpatchify(x1, nh=12, nw=16)
        plt.figure()
        plt.imshow(x_grid)
        plt.show()
        break