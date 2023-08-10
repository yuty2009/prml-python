
import os
import glob
import torch
import numpy as np
from PIL import Image


subsets = {'40X', '100X', '200X', '400X'}
label2index = {'B_': 0, 'M_': 1}

class BREAKHISPatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', subset='40X', patch_size=28, transform=None, transform_patch=None):
        self.root = root
        self.subset = subset
        self.patch_size = patch_size
        self.transform = transform
        self.transform_patch = transform_patch
        self.classes = ['B_', 'M_']
        self.class_to_idx = label2index
        self.imgs = self.make_dataset(split)
        
    def make_dataset(self, split='train'):
        imgs = []
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(self.root, split, self.subset, target+'*/*.png')
            img_list = glob.glob(d)
            for path in img_list:
                imgs.append((path, self.class_to_idx[target]))
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
    
    root = 'f:/medicalimages/breakhis'
    tf_resize = transforms.CenterCrop((448, 700))
    tf_train = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = torchvision.datasets.ImageFolder(root, transform)
    dataset = BREAKHISPatchDataset(root, split='test', patch_size=28, transform=tf_resize, transform_patch=tf_train)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        x1 = x[0].numpy().transpose(0, 2, 3, 1)
        x_grid = dataset.unpatchify(x1, nh=16, nw=25)
        plt.figure()
        plt.imshow(x_grid)
        plt.show()
        break