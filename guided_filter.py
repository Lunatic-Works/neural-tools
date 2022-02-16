#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import uniform_filter

from utils import do_imgs, read_img, write_img

in_filenames = [
    './in.png',
]
out_suffix = '_gf'

output_8_bit = False


def box_filter(x, d):
    if x.ndim == 2:
        return uniform_filter(x, d)

    assert x.ndim == 3
    out = np.empty_like(x)
    for i in range(x.shape[2]):
        uniform_filter(x[:, :, i], d, out[:, :, i])
    return out


def guided_filter(x, y, d, eps):
    assert x.shape == y.shape

    mean_x = box_filter(x, d)
    mean_y = box_filter(y, d)
    cov_xy = box_filter(x * y, d) - mean_x * mean_y
    var_x = box_filter(x * x, d) - mean_x**2

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, d)
    mean_b = box_filter(b, d)

    output = mean_A * x + mean_b
    return output


def convert_img(_, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=True, signed=False)

    img = guided_filter(img, img, 3, 1e-3)

    write_img(out_filename,
              img,
              swap_rb=True,
              signed=False,
              output_8_bit=output_8_bit)


if __name__ == '__main__':
    do_imgs(convert_img,
            None,
            in_filenames,
            out_suffix,
            out_extname=None if output_8_bit else '.png')
