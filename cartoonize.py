#!/usr/bin/env python3

import numpy as np
import skimage.transform

from utils import do_imgs, get_batch, get_pieces, merge_img, read_img, write_img

model_filenames = [
    "./models/cartoonize/shinkai.onnx",
]
in_filenames = [
    "./in.png",
]

piece_inner_size = 240
pad_size = 60
batch_size = 8

swap_rb = True
noise = 0.01
scale = 1
output_8_bit = False


def convert_img(sess, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=swap_rb, signed=True, noise=noise)

    if scale != 1:
        img = skimage.transform.rescale(img, scale, channel_axis=2)

    # Use the whole image to calibrate InstanceNorm
    piece_outer_size = piece_inner_size + pad_size * 2
    img_small = skimage.transform.resize(img, (piece_outer_size,) * 2)
    img_small = img_small[None, :, :, :].transpose(0, 3, 1, 2)

    pieces, max_row_col, pads = get_pieces(img, piece_inner_size, pad_size)

    out_pieces = []
    for batch in get_batch(pieces, batch_size):
        batch = np.concatenate([img_small, batch])
        out_batch = sess.run(None, {"in": batch})[0]
        out_batch = out_batch[1:]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_pieces.append(out_batch)
    out_pieces = np.concatenate(out_pieces)

    out_img = merge_img(out_pieces, piece_inner_size, pad_size, max_row_col, pads)

    write_img(
        out_filename, out_img, swap_rb=swap_rb, signed=True, output_8_bit=output_8_bit
    )


if __name__ == "__main__":
    do_imgs(
        convert_img,
        model_filenames,
        in_filenames,
        out_extname=None if output_8_bit else ".png",
    )
