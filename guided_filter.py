#!/usr/bin/env python3

from cv2.ximgproc import guidedFilter

from utils import do_imgs, read_img, write_img

in_filenames = [
    "./in.png",
]
out_suffix = "_gf"

output_8_bit = False


def convert_img(_, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=True, signed=False)

    img = guidedFilter(img, img, 3, 1e-3)

    write_img(out_filename, img, swap_rb=True, signed=False, output_8_bit=output_8_bit)


if __name__ == "__main__":
    do_imgs(
        convert_img,
        None,
        in_filenames,
        out_suffix,
        out_extname=None if output_8_bit else ".png",
    )
