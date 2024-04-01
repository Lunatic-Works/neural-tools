#!/usr/bin/env python3

from utils import do_imgs, read_img, write_img

in_filenames = [
    "./in.png",
]
out_suffix = "_quant"


def convert_img(_, in_filename, out_filename):
    img = read_img(in_filename, signed=False)
    write_img(out_filename, img, signed=False, quant_bit="adapt")


if __name__ == "__main__":
    do_imgs(convert_img, None, in_filenames, out_suffix=out_suffix, out_extname=".png")
