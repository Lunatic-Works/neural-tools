#!/usr/bin/env python3

import numpy as np
import skimage.transform

from utils import do_imgs, read_img, write_img

model_filename = './models/stylize_2/cezanne.onnx'
in_filenames = [
    './in.png',
]
out_suffix = '_cezanne_2'

size = 2048

swap_rb = True
noise = 0
output_8_bit = False


def convert_img(sess, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=swap_rb, signed=False, noise=noise)
    original_size = img.shape[:2]
    img = skimage.transform.resize(img, (size, size))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    out_img = sess.run(None, {'Input': img})[0]

    out_img = out_img.squeeze(axis=0)
    out_img = out_img.transpose(1, 2, 0)
    out_img = skimage.transform.resize(out_img, original_size)
    write_img(out_filename,
              out_img,
              swap_rb=swap_rb,
              signed=False,
              output_8_bit=output_8_bit)


if __name__ == '__main__':
    do_imgs(convert_img,
            model_filename,
            in_filenames,
            out_suffix,
            out_extname=None if output_8_bit else '.png')
