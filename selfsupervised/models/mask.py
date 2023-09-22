
import math
import random
import numpy as np


class MaskGenerator2d:
    # following iBot (https://arxiv.org/abs/2111.07832) and
    # BEiT (https://arxiv.org/abs/2106.08254), see at
    # https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
    def __init__(self, mask_prob=0.6, min_aspect=0.3, mask_type='block'):
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        max_aspect = max(1 / min_aspect, min_aspect)
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(self, input_shape):
        if len(input_shape) == 2:
            batch_size, sequence_length = input_shape
            h_patches = w_patches = int(math.sqrt(sequence_length))
        elif len(input_shape) == 3:
            batch_size, h_patches, w_patches = input_shape
        else:
            raise NotImplementedError
        high = int(h_patches * w_patches * self.mask_prob)
        low = (min(h_patches, w_patches) // 3) ** 2

        masks = []
        for it in range(batch_size):
            if self.mask_type == 'block':
                mask = np.zeros((h_patches, w_patches), dtype=int)
                mask_count = 0
                while mask_count < high:
                    mask_count = high - mask_count
                    delta = self.mask_block(mask, h_patches, w_patches, low, high)
                    if delta == 0:
                        break
                    mask_count += delta
            elif self.mask_type == 'random':
                mask = np.hstack([
                    np.zeros(h_patches * w_patches - high),
                    np.ones(high),
                ]).astype(int)
                np.random.shuffle(mask)
                mask = mask.reshape((h_patches, w_patches))
            else:
                raise NotImplementedError
            masks.append(mask)

        masks = np.stack(masks)
        return masks

    def mask_block(self, mask, h_patches, w_patches, min_mask_patches, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(min_mask_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < w_patches and h < h_patches:
                top = random.randint(0, h_patches - h)
                left = random.randint(0, w_patches - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta
    

class MaskGenerator1d:
    # block mask follows the implementation of PatchMaskGenerator
    # random mask follows the implementation of HuBERT, see at
    # https://github.com/bshall/hubert/blob/main/hubert/model.py
    def __init__(self, mask_prob=0.65, mask_length=10, min_masks=1, mask_type='default'):
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        assert mask_length >= 1, "`mask_length` has to be bigger than 0."
        self.mask_length = mask_length
        self.min_masks = min_masks

    def __call__(self, input_shape):
        batch_size, sequence_length = input_shape
        assert self.mask_length <= sequence_length, \
            f"`mask_length` has to be smaller than `sequence_length`, " + \
            "but got `mask_length`: {self.mask_length} and `sequence_length`: {sequence_length}`"
        # SpecAugment mask to fill
        masks = np.zeros((batch_size, sequence_length), dtype=int)
        if self.mask_type == "block":
            high = int(sequence_length * self.mask_prob)
            low = high // 2 # sequence_length // 3
            # `mask_length` and `min_masks` are not used in this case
            mask_length = int(random.uniform(low, high))
            for i in range(batch_size):
                start = random.randint(0, sequence_length - mask_length)
                end = start + mask_length
                masks[i, start:end] = 1
        # ensure masking the same number of tokens in each sequence of the batch
        elif self.mask_type == "batch":
            high = int(sequence_length * self.mask_prob)
            for i in range(batch_size):
                mask = np.hstack([
                    np.zeros(sequence_length - high),
                    np.ones(high),
                ]).astype(int)
                np.random.shuffle(mask)
                masks[i, :] = mask
        # masking multiple blocks of continuous tokens starting at random positions
        elif self.mask_type == "random":
            mask_prob = self.mask_prob / self.mask_length
            start_pos = np.random.rand(batch_size, sequence_length) < mask_prob
            n_starts = start_pos.sum()
            start_ind = np.where(start_pos)
            for k in range(n_starts):
                i = start_ind[0][k]
                j = start_ind[1][k]
                masks[i, j:j + self.mask_length] = 1
        elif self.mask_type == "random1":
            # compute number of masked spans in batch
            num_masked_spans = int(self.mask_prob * sequence_length / self.mask_length + random.random())
            num_masked_spans = max(num_masked_spans, self.min_masks)

            # make sure num masked indices <= sequence_length
            if num_masked_spans * self.mask_length > sequence_length:
                num_masked_spans = sequence_length // self.mask_length

            # uniform distribution to sample from, make sure that offset samples are < sequence_length
            uniform_seq = [
                np.random.permutation(sequence_length - (self.mask_length - 1))
                for i in range(batch_size)
            ]
            uniform_seq = np.stack(uniform_seq)

            # get random indices to mask
            mask_indices = uniform_seq[:, :num_masked_spans]

            # expand masked indices to masked spans
            mask_indices = (
                mask_indices[:, :, np.newaxis]
                .repeat(self.mask_length, axis=-1)
                .reshape(batch_size, num_masked_spans * self.mask_length)
            )
            offsets = (
                np.arange(self.mask_length)[np.newaxis, np.newaxis, :]
                .repeat(batch_size, axis=0).repeat(num_masked_spans, axis=1)
                .reshape(batch_size, num_masked_spans * self.mask_length)
            )
            mask_idxs = mask_indices + offsets

            # scatter indices to mask
            for i in range(batch_size):
                masks[i, mask_idxs[i]] = 1
        
        return masks
    

if __name__ == '__main__':

    from PIL import Image
    import matplotlib.pyplot as plt

    
    img = Image.open('/Users/yuty2009/data/prmldata/imagenet-1k/train/n01440764/n01440764_18.JPEG')
    img = img.resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1)
    mask = MaskGenerator2d(mask_prob=0.75, mask_type='block')
    mask = mask((1, *img_data.shape[1:]))
    
    img_masked = img_data.copy()
    img_masked[:, mask[0] == 1] = 0
    img_masked = img_masked.transpose(1, 2, 0)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_masked)
    plt.show()
    """

    fs = 250.0
    t = np.linspace(0, 4.0, int(4.0 * fs), endpoint=False)
    x = np.sin(2 * np.pi * 30 * np.sqrt(t))

    mask = MaskGenerator1d(mask_prob=0.8, mask_length=10, min_masks=2, mask_type='block')
    mask = mask((1, len(x)))
    x_masked = x.copy()
    x_masked[mask[0] == 1] = 0

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.subplot(2, 1, 2)
    plt.plot(t, x_masked)
    plt.show()
    """
