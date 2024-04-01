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
out_suffix = None

piece_inner_size = 240
pad_size = 60
batch_size = 8

swap_rb = False
noise = 0.01
scale = None
run_size = None
wrap_x = False
wrap_y = False
output_png = True
output_8_bit = False


def convert_img(sess, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=swap_rb, signed=True, noise=noise)

    if scale:
        img = skimage.transform.rescale(img, scale, channel_axis=2)

    if run_size:
        original_shape = img.shape[:2]
        run_scale = max(run_size / img.shape[0], run_size / img.shape[1])
        print("run_scale", run_scale)
        img = skimage.transform.rescale(img, run_scale, channel_axis=2)

    pieces, max_row_col, pads = get_pieces(
        img, piece_inner_size, pad_size, wrap_x=wrap_x, wrap_y=wrap_y
    )

    out_pieces = []
    for batch in get_batch(pieces, batch_size):
        out_batch = sess.run(None, {"in": batch})[0]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_pieces.append(out_batch)
    out_pieces = np.concatenate(out_pieces)

    out_img = merge_img(out_pieces, piece_inner_size, pad_size, max_row_col, pads)

    if run_size:
        out_img = skimage.transform.resize(out_img, original_shape)

    write_img(
        out_filename, out_img, swap_rb=swap_rb, signed=True, output_8_bit=output_8_bit
    )


if __name__ == "__main__":
    do_imgs(
        convert_img,
        model_filenames,
        in_filenames,
        out_suffix=out_suffix,
        out_extname=".png" if output_png or not output_8_bit else None,
    )
