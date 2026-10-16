#!/usr/bin/env python3

import numpy as np
from scipy import ndimage
from skimage import color, filters, morphology

from utils import do_imgs, read_img, write_img

in_filenames = [
    "./in.png",
]
out_suffix = "_cr"

line_sigmas = [0.5, 1, 1.5, 2]
eps = 1e-7
output_png = True
output_8_bit = False


def convert_img(_, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=True, signed=False)

    gray = color.rgb2gray(img)
    ridges = filters.frangi(gray, sigmas=line_sigmas, black_ridges=True)

    ridges_filename = out_filename.replace("_cr", "_cr_ridges")
    write_img(ridges_filename, ridges, signed=False, output_8_bit=output_8_bit)

    low_val = np.percentile(ridges, 98)
    high_val = np.percentile(ridges, 99)
    mask = filters.apply_hysteresis_threshold(ridges, low_val, high_val)
    # mask = morphology.remove_small_objects(mask, max_size=10)

    mask_filename = out_filename.replace("_cr", "_cr_mask")
    write_img(mask_filename, mask, signed=False, output_8_bit=output_8_bit)

    mask_dilated = morphology.dilation(mask, morphology.disk(1))
    indices = ndimage.distance_transform_edt(
        mask_dilated, return_indices=True, return_distances=False
    )
    out = img[indices[0], indices[1], :]

    write_img(out_filename, out, swap_rb=True, signed=False, output_8_bit=output_8_bit)

    mul = np.clip(img / np.maximum(out, eps), 0, 1)

    mul_filename = out_filename.replace("_cr", "_cr_mul")
    write_img(mul_filename, mul, swap_rb=True, signed=False, output_8_bit=output_8_bit)


if __name__ == "__main__":
    do_imgs(
        convert_img,
        None,
        in_filenames,
        out_suffix=out_suffix,
        out_extname=".png" if output_png or not output_8_bit else None,
    )
