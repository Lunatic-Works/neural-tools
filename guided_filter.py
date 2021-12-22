#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import uniform_filter

from utils import do_imgs, read_img, write_img

in_filenames = [
    './in.png',
]
out_suffix = '_gf'

output_8_bit = False


def box_filter(x, r):
    return np.stack([uniform_filter(x[:, :, i], r) for i in range(x.shape[2])],
                    axis=2)


def guided_filter(x, y, r, eps):
    assert x.shape == y.shape

    mean_x = box_filter(x, r)
    mean_y = box_filter(y, r)
    cov_xy = box_filter(x * y, r) - mean_x * mean_y
    var_x = box_filter(x * x, r) - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r)
    mean_b = box_filter(b, r)

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
