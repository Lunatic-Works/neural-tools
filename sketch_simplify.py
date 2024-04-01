#!/usr/bin/env python3

import numpy as np
import skimage.color

from utils import do_imgs, get_batch, get_pieces, merge_img, read_img, write_img

model_filename = "./models/sketch_simplify/sketch_gan.onnx"
in_filenames = [
    "./in.png",
]
out_suffix = "_simplify"

piece_inner_size = 144
pad_size = 24
batch_size = 135

gain = 10
output_8_bit = False


def convert_img(sess, in_filename, out_filename):
    img = read_img(in_filename, signed=False)
    img = skimage.color.rgb2gray(img)
    img = img[:, :, None]
    img = (img - 1) * gain + 1

    pieces, max_row_col, pads = get_pieces(img, piece_inner_size, pad_size)

    out_pieces = []
    for batch in get_batch(pieces, batch_size):
        out_batch = sess.run(None, {"in": batch})[0]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_pieces.append(out_batch)
    out_pieces = np.concatenate(out_pieces)

    out_img = merge_img(out_pieces, piece_inner_size, pad_size, max_row_col, pads)

    write_img(out_filename, out_img, signed=False, output_8_bit=output_8_bit)


if __name__ == "__main__":
    do_imgs(
        convert_img,
        model_filename,
        in_filenames,
        out_suffix=out_suffix,
        out_extname=None if output_8_bit else ".png",
    )
