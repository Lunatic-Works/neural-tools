#!/usr/bin/env python3

import numpy as np
import skimage.transform

from utils import (do_imgs, floor_even, get_batch, get_pieces, merge_img,
                   read_img, trim_img, untrim_img, write_img)

model_filename = './models/waifu2x/anime/noise1_scale2x.onnx'
in_filenames = [
    './in.png',
]
out_suffix = '_waifu2x'

piece_inner_size = 36
pad_size = 12
up_scale = 2
up_shift = 14
batch_size = 400

trim_alpha = False
trim_eps = 1e-3
upscale = True
downscale = True
run_alpha = False
alpha_gamma = 1
output_gray = False
output_alpha = False
output_8_bit = True

if trim_alpha or run_alpha:
    assert output_alpha


def run_img(sess, img):
    pieces, max_row_col, pads = get_pieces(img, piece_inner_size, pad_size)

    out_pieces = []
    for batch in get_batch(pieces, batch_size):
        out_batch = sess.run(None, {'in': batch})[0]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_pieces.append(out_batch)
    out_pieces = np.concatenate(out_pieces)

    out_img = merge_img(out_pieces, piece_inner_size, pad_size, max_row_col,
                        pads, (up_scale, up_shift))
    out_img = np.clip(out_img, 0, 1)

    return out_img


def convert_img(sess, in_filename, out_filename):
    # Network input is BGR
    img, alpha = read_img(in_filename,
                          swap_rb=False,
                          signed=False,
                          return_alpha=True)

    if trim_alpha and alpha is not None:
        original_shape, (trim_t, trim_b, trim_l,
                         trim_r) = trim_img(img, alpha, trim_eps)

        if not (upscale and not downscale):
            trim_b = trim_t + floor_even(trim_b - trim_t)
            trim_r = trim_l + floor_even(trim_r - trim_l)

        img = img[trim_t:trim_b, trim_l:trim_r, :]
        alpha = alpha[trim_t:trim_b, trim_l:trim_r]

        if upscale and not downscale:
            original_shape = (original_shape[0] * 2, original_shape[1] * 2)
            trim_t *= 2
            trim_b *= 2
            trim_l *= 2
            trim_r *= 2

    if upscale:
        if not run_alpha and alpha is not None:
            if downscale:
                alpha = skimage.transform.resize(alpha,
                                                 floor_even(alpha.shape))
            else:
                alpha = skimage.transform.resize(
                    alpha, (alpha.shape[0] * 2, alpha.shape[1] * 2))
    else:
        img = skimage.transform.resize(img,
                                       (img.shape[0] // 2, img.shape[1] // 2))
        if alpha is not None:
            if run_alpha:
                alpha = skimage.transform.resize(
                    alpha, (alpha.shape[0] // 2, alpha.shape[1] // 2))
            else:
                alpha = skimage.transform.resize(alpha,
                                                 floor_even(alpha.shape))

    img = run_img(sess, img)

    if run_alpha:
        alpha = np.repeat(alpha[:, :, None], 3, axis=2)
        alpha = run_img(sess, alpha)
        alpha = alpha.mean(axis=2)
        alpha = np.clip(alpha, 0, 1)
        alpha[alpha < 1 - trim_eps] **= alpha_gamma

    if upscale and downscale:
        img = skimage.transform.resize(img,
                                       (img.shape[0] // 2, img.shape[1] // 2))
        if run_alpha:
            alpha = skimage.transform.resize(
                alpha, (alpha.shape[0] // 2, alpha.shape[1] // 2))

    if trim_alpha and alpha is not None:
        img, alpha = untrim_img(img, alpha, original_shape,
                                (trim_t, trim_b, trim_l, trim_r))

    # Network output is BGR
    write_img(out_filename,
              img,
              alpha if output_alpha else None,
              swap_rb=False,
              signed=False,
              output_gray=output_gray,
              output_8_bit=output_8_bit)


if __name__ == '__main__':
    do_imgs(convert_img,
            model_filename,
            in_filenames,
            out_suffix,
            out_extname=None if output_8_bit else '.png')
