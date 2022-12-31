#!/usr/bin/env python3

import numpy as np
import skimage.transform

from guided_filter import box_filter
from rtv_smooth import rtv_smooth, tv_smooth
from utils import do_imgs, read_img, write_img

in_filenames = [
    "./in.png",
]
out_suffix = "_simplify"

scale = 0.25
erase_ratio = 0.5
max_iter = 5
output_8_bit = False


def _write_img(filename, img):
    write_img(
        filename,
        img,
        swap_rb=(img.ndim == 3),
        signed=False,
        output_8_bit=output_8_bit or img.ndim != 3,
    )


def convert_img(sess, in_filename, out_filename, eps=1e-15):
    img = read_img(in_filename, swap_rb=True, signed=False)

    img = skimage.transform.rescale(img, scale, channel_axis=2)

    height, width, _ = img.shape
    mask = np.random.rand(height, width) < 0.5
    for i in range(max_iter):
        print("eap", i)

        img_new = tv_smooth(img, mask)
        img_new = rtv_smooth(img_new)

        filename = out_filename.replace(out_suffix, f"{out_suffix}_mask{i}")
        _write_img(filename, mask.astype(np.float64))
        filename = out_filename.replace(out_suffix, f"{out_suffix}_out{i}")
        _write_img(filename, img_new)

        if i == max_iter - 1:
            break

        value = ((img_new - img) ** 2).sum(axis=2)
        weight = ((img_new - box_filter(img_new, 3)) ** 2).sum(axis=2)
        knapsack = value / (weight + eps)
        threshold = np.quantile(knapsack, 1 - (i + 1) / (max_iter - 1) * erase_ratio)
        knapsack /= threshold + eps
        mask = np.random.rand(height, width) < knapsack


if __name__ == "__main__":
    do_imgs(
        convert_img,
        None,
        in_filenames,
        out_suffix,
        out_extname=None if output_8_bit else ".png",
    )
