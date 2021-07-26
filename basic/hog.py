# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog


def hog_feature(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                transform_sqrt=False, block_norm='L2-Hys'):

    image = image.astype(np.float, copy=False)

    # The first stage either compute the square root or the log of each color channel.
    if transform_sqrt:
        image = np.sqrt(image)

    # The second stage computes first order image gradients.
    if image.ndim == 2:
        g_row, g_col = _hog_channel(image)
        # g_magn = np.sqrt(g_row**2 + g_col**2)
    elif image.ndim == 3:
        g_row_by_ch = np.empty_like(image, dtype=np.float)
        g_col_by_ch = np.empty_like(image, dtype=np.float)
        g_magn = np.empty_like(image, dtype=np.float)
        for i in range(image.shape[2]):
            g_row_by_ch[:, :, i], g_col_by_ch[:, :, i] = _hog_channel(image[:,:,i])
            # np.hypot is equivalent to sqrt(x1**2 + x2**2), element-wise.
            g_magn[:, :, i] = np.hypot(g_row_by_ch[:, :, i], g_col_by_ch[:, :, i])
            # For each pixel select the channel with the highest gradient magnitude
            idcs_max = g_magn.argmax(axis=2)
            rr, cc = np.meshgrid(np.arange(image.shape[0]),
                                np.arange(image.shape[1]),
                                indexing='ij',
                                sparse=True)
            g_row = g_row_by_ch[rr, cc, idcs_max]
            g_col = g_col_by_ch[rr, cc, idcs_max]

    # The third stage aims to produce an encoding that is sensitive to local image
    # content while remaining resistant to small changes in pose or appearance.
    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block
    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis
    g_row = g_row.astype(np.float, copy=False)
    g_col = g_col.astype(np.float, copy=False)
    orient_hist = _hog_histgram(g_col, g_row, c_col, c_row, 
                                  n_cells_col, n_cells_row, orientations)
    hog_image = _hog_image(orient_hist, c_col, c_row, s_col, s_row, 
                           n_cells_col, n_cells_row, orientations)

    # The fourth stage computes normalization, which takes local groups of cells and 
    # contrast normalizes their overall responses before passing to next stage.
    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, orientations),
        dtype=np.float
    )

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orient_hist[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = _hog_normalize_block(block, block_norm)

    return normalized_blocks.ravel(), hog_image

def _hog_image(orient_hist, c_col, c_row, s_col, s_row, 
               n_cells_col, n_cells_row, orientations):
    from skimage.draw import line

    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col), dtype=np.float)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = line(int(centre[0] - dc),
                                    int(centre[1] + dr),
                                    int(centre[0] + dc),
                                    int(centre[1] - dr))
                hog_image[rr, cc] += orient_hist[r, c, o]

    return hog_image

def _hog_channel(image):
    g_row = np.zeros(image.shape, dtype=image.dtype)
    g_col = np.zeros(image.shape, dtype=image.dtype)
    g_row[1:-1, :] = sig.convolve2d(image, [[1], [0], [-1]], mode='valid')
    g_col[:, 1:-1] = sig.convolve2d(image, [[1, 0, -1]], mode='valid')
    # skimage.feature.hog uses the following code 
    """
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    """
    return g_row, g_col

def _hog_histgram(g_col, g_row, c_col, c_row, n_cells_col, n_cells_row, orientations):

    g_magn = np.hypot(g_col, g_row)
    g_orient = np.rad2deg(np.arctan2(g_row, g_col)) % 180

    orient_hist = np.zeros((n_cells_row, n_cells_col, orientations), dtype=float)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            g_magn_cell = g_magn[r*c_row:(r+1)*c_row, c*c_col:(c+1)*c_col]
            g_orient_cell = g_orient[r*c_row:(r+1)*c_row, c*c_col:(c+1)*c_col]
            orient_hist[r, c, :] = _hog_histgram_cell(g_magn_cell, g_orient_cell,
                                                      orientations)

    return orient_hist

def _hog_histgram_cell(g_magn_cell, g_orient_cell, orientations):
    bin_size = 180.0 / orientations
    c_row, c_col = g_magn_cell.shape
    orient_hist_cell = np.zeros(orientations)
    for r in range(c_row):
        for c in range(c_col):
            o = g_orient_cell[r, c]
            orient_idx = int(o / bin_size)
            orient_hist_cell[orient_idx] += g_magn_cell[r, c]
    return orient_hist_cell / (c_row * c_col)

def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')
    return out

  
if __name__ == "__main__":

    from PIL import Image
    import matplotlib.pyplot as plt

    img = imread('.data/cat.jpg')
    img1 = Image.open('.data/cat.jpg')
    img_resized = resize(img, (128*4, 64*4))
    fd, img_hog = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd1, img_hog1 = hog_feature(img_resized)

    plt.figure(1)
    ax = plt.subplot(221)
    ax.axis("off")
    ax.imshow(img_resized)
    ax = plt.subplot(222)
    ax.axis("off")
    ax.imshow(img_hog, cmap="gray")
    ax = plt.subplot(224)
    ax.axis("off")
    ax.imshow(img_hog1, cmap="gray")
    plt.show()