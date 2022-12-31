#!/usr/bin/env python3

import skimage.color
from skimage.exposure import equalize_adapthist
from skimage.filters import difference_of_gaussians

from utils import do_imgs, read_img, write_img

in_filenames = [
    "./in.png",
]
out_suffix = "_hf"

radius = 10
output_8_bit = False


def convert_img(_, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=True, signed=False)
    img = skimage.color.rgb2gray(img)

    img = equalize_adapthist(img, kernel_size=radius)

    img = difference_of_gaussians(img, 0, radius)
    img = (img + 1) / 2

    write_img(out_filename, img, signed=False, output_8_bit=output_8_bit)


if __name__ == "__main__":
    do_imgs(
        convert_img,
        None,
        in_filenames,
        out_suffix,
        out_extname=None if output_8_bit else ".png",
    )
