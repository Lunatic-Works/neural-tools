#!/usr/bin/env python3

import numpy as np
import skimage.transform

from utils import do_imgs, floor_even, read_img, trim_img, untrim_img, write_img
from waifu2x import run_img

model_filename = "./models/waifu2x/anime/noise1_scale2x.onnx"
in_filenames = [
    "./in.png",
]
out_suffix = "_waifu2x"

trim_eps = 1e-3
alpha_pad = 0.05


def convert_img(sess, in_filename, out_filename):
    # Network input is BGR
    img, alpha = read_img(in_filename, swap_rb=False, signed=False, return_alpha=True)

    original_shape, (trim_t, trim_b, trim_l, trim_r) = trim_img(img, alpha, trim_eps)
    trim_b = trim_t + floor_even(trim_b - trim_t)
    trim_r = trim_l + floor_even(trim_r - trim_l)
    img = img[trim_t:trim_b, trim_l:trim_r, :]
    alpha = alpha[trim_t:trim_b, trim_l:trim_r]

    img = skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    alpha = skimage.transform.resize(alpha, (alpha.shape[0] // 2, alpha.shape[1] // 2))
    alpha = alpha[:, :, None]
    img_black = img * alpha
    img_white = img * alpha + 1 - alpha

    img_black_out = run_img(sess, img_black)
    img_white_out = run_img(sess, img_white)
    alpha_out = (1 + img_black_out - img_white_out).mean(axis=2)
    alpha_out = alpha_out * (1 + alpha_pad * 2) - alpha_pad
    alpha_out = np.clip(alpha_out, 0, 1)
    img_out = img_black_out * (1 + trim_eps) / (alpha_out[:, :, None] + trim_eps)
    img_out = np.clip(img_out, 0, 1)

    img, alpha = untrim_img(
        img_out, alpha_out, original_shape, (trim_t, trim_b, trim_l, trim_r)
    )

    # Network output is BGR
    write_img(out_filename, img, alpha, swap_rb=False, signed=False)


if __name__ == "__main__":
    do_imgs(convert_img, model_filename, in_filenames, out_suffix)
