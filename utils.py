import os
import shutil
from math import ceil, floor

import cv2
import numpy as np
import onnxruntime as rt
import skimage


def floor_even(x):
    if isinstance(x, tuple):
        return tuple(floor_even(y) for y in x)
    if isinstance(x, list):
        return [floor_even(y) for y in x]
    return x // 2 * 2


# Does not copy img and alpha
def trim_img(img, alpha, eps):
    original_shape = alpha.shape

    trim_t = 0
    while np.all(alpha[trim_t, :] < eps):
        trim_t += 1
    trim_b = alpha.shape[0] - 1
    while np.all(alpha[trim_b, :] < eps):
        trim_b -= 1
    trim_l = 0
    while np.all(alpha[:, trim_l] < eps):
        trim_l += 1
    trim_r = alpha.shape[1] - 1
    while np.all(alpha[:, trim_r] < eps):
        trim_r -= 1

    trim_b += 1
    trim_r += 1

    trims = (trim_t, trim_b, trim_l, trim_r)
    return original_shape, trims


def untrim_img(img, alpha, original_shape, trims):
    trim_t, trim_b, trim_l, trim_r = trims
    new_img = np.zeros([original_shape[0], original_shape[1], 3])
    new_alpha = np.zeros(original_shape)
    new_img[trim_t:trim_b, trim_l:trim_r, :] = img
    new_alpha[trim_t:trim_b, trim_l:trim_r] = alpha
    return new_img, new_alpha


def get_pieces(img, piece_inner_size, pad_size):
    piece_outer_size = piece_inner_size + pad_size * 2

    max_row = ceil(img.shape[0] / piece_inner_size)
    max_col = ceil(img.shape[1] / piece_inner_size)
    img_padded_h = max_row * piece_inner_size
    img_padded_w = max_col * piece_inner_size
    pad_t = floor((img_padded_h - img.shape[0]) / 2)
    pad_b = img_padded_h - img.shape[0] - pad_t
    pad_l = floor((img_padded_w - img.shape[1]) / 2)
    pad_r = img_padded_w - img.shape[1] - pad_l
    img_full = np.pad(img, [
        (pad_t + pad_size, pad_b + pad_size),
        (pad_l + pad_size, pad_r + pad_size),
        (0, 0),
    ], 'reflect')

    pieces = []
    for i in range(max_row):
        for j in range(max_col):
            idx_t = i * piece_inner_size
            idx_b = idx_t + piece_outer_size
            idx_l = j * piece_inner_size
            idx_r = idx_l + piece_outer_size
            pieces.append(img_full[idx_t:idx_b, idx_l:idx_r, :])
    pieces = np.stack(pieces)

    max_row_col = (max_row, max_col)
    pads = (pad_t, pad_b, pad_l, pad_r)
    return pieces, max_row_col, pads


def get_batch(pieces, batch_size):
    idx = 0
    while idx < pieces.shape[0]:
        batch = pieces[idx:idx + batch_size]
        batch = batch.transpose(0, 3, 1, 2)
        idx += batch.shape[0]
        print('Piece {}/{}'.format(idx, pieces.shape[0]))
        yield batch


def merge_img(pieces,
              piece_inner_size,
              pad_size,
              max_row_col,
              pads,
              scale_shift=(1, 0)):
    max_row, max_col = max_row_col
    pad_t, pad_b, pad_l, pad_r = pads
    scale, shift = scale_shift

    piece_outer_size = piece_inner_size + pad_size * 2
    scaled_inner_size = piece_inner_size * scale
    scaled_outer_size = piece_outer_size * scale

    img = np.empty(
        (max_row * scaled_outer_size, max_col * scaled_outer_size, 3))
    for idx, piece in enumerate(pieces):
        i = idx // max_col
        j = idx % max_col

        idx_t = i * scaled_inner_size
        idx_b = idx_t + scaled_inner_size
        idx_l = j * scaled_inner_size
        idx_r = idx_l + scaled_inner_size
        piece_l = pad_size * scale - shift
        piece_r = piece_l + scaled_inner_size
        img[idx_t:idx_b, idx_l:idx_r, :] = piece[piece_l:piece_r,
                                                 piece_l:piece_r, :]

    img = img[pad_t * scale:(max_row * piece_inner_size - pad_b) * scale,
              pad_l * scale:(max_col * piece_inner_size - pad_r) * scale, :]
    return img


def read_img(filename,
             swap_rb=False,
             gamma=1,
             signed=True,
             scale=None,
             return_alpha=False):
    # Use cv2 to support 16 bit image
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = skimage.img_as_float32(img)

    alpha = None

    if img.ndim == 3:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
        else:
            assert img.shape[2] == 3
        # Remove alpha channel
        img = img[:, :, :3]
    else:
        assert img.ndim == 2
        # Convert grayscale to RGB
        img = np.repeat(img[:, :, None], 3, axis=2)

    if swap_rb:
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]

    img **= gamma

    if signed:
        # [0, 1] -> [-1, 1]
        img = img * 2 - 1

    if scale is not None:
        img *= scale
        if alpha is not None:
            alpha *= scale

    if return_alpha:
        return img, alpha
    else:
        return img


def write_img(filename,
              img,
              alpha=None,
              swap_rb=False,
              signed=True,
              scale=None,
              output_gray=False,
              output_8_bit=True):
    if scale is not None:
        img /= scale
        if alpha is not None:
            alpha /= scale

    if signed:
        # [-1, 1] -> [0, 1]
        img = (img + 1) / 2

    img = np.clip(img, 0, 1)

    if swap_rb:
        # RGB -> BGR
        img = img[:, :, [2, 1, 0]]

    if output_gray:
        img = img.mean(axis=2, keepdims=True)

    if alpha is not None:
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        img = np.concatenate([img, alpha], axis=2)

    if output_8_bit:
        img = skimage.img_as_ubyte(img)
    else:
        img = skimage.img_as_uint(img)

    cv2.imwrite(filename, img)


def do_imgs(fun,
            model_filename,
            in_filenames,
            out_suffix,
            out_extname=None,
            tmp_filename=None):
    sess = rt.InferenceSession(model_filename)

    for in_filename in in_filenames:
        print(in_filename)

        basename, extname = os.path.splitext(in_filename)
        if out_extname:
            out_filename = basename + out_suffix + out_extname
        else:
            out_filename = basename + out_suffix + extname

        if tmp_filename:
            shutil.copy2(in_filename, tmp_filename)
            fun(sess, tmp_filename, tmp_filename)
            shutil.move(tmp_filename, out_filename)
        else:
            fun(sess, in_filename, out_filename)
