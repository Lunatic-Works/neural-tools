#!/usr/bin/env python3

import glob
import os

import numpy as np
import skimage.transform

from utils import (
    create_session,
    get_batch,
    get_tiles,
    merge_img,
    read_img,
    write_img,
)

style = "shinkai"
preview_model_filename = f"./models/cartoonize_twopass/{style}_preview.onnx"
upscale_model_filename = f"./models/cartoonize_twopass/{style}_upscale.onnx"
in_filenames = [
    "./in.png",
]
out_suffix = f"_{style}"
# out_suffix = f"_{style}_rb"

preview_size = 512
preview_min_side = 64

tile_inner_size = 336
pad_size = 72
batch_size = 8

swap_rb = False
# swap_rb = True
noise = 0.01
scale = None
run_size = None
wrap_x = False
wrap_y = False
output_png = True
output_8_bit = False


def compute_stats(preview_session, img):
    h, w = img.shape[:2]
    scale = min(1.0, preview_size / max(h, w))
    ph = max(preview_min_side, int(round(h * scale)) // 4 * 4)
    pw = max(preview_min_side, int(round(w * scale)) // 4 * 4)
    if (ph, pw) != (h, w):
        small = skimage.transform.resize(img, (ph, pw))
    else:
        small = img
    x = np.ascontiguousarray(small.transpose(2, 0, 1)[None]).astype(np.float32)
    return preview_session.run(None, {"in": x})


def run_img(preview_session, upscale_session, img):
    stats = compute_stats(preview_session, img)

    tiles, max_row_col, pads = get_tiles(
        img, tile_inner_size, pad_size, wrap_x=wrap_x, wrap_y=wrap_y
    )

    stat_names = [i.name for i in upscale_session.get_inputs()][1:]
    out_tiles = []
    for batch in get_batch(tiles, batch_size):
        feed = {"in": batch}
        for name, stat in zip(stat_names, stats):
            feed[name] = stat
        out_batch = upscale_session.run(None, feed)[0]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_tiles.append(out_batch)
    out_tiles = np.concatenate(out_tiles)

    out_img = merge_img(out_tiles, tile_inner_size, pad_size, max_row_col, pads)
    return out_img


def convert_img(preview_session, upscale_session, in_filename, out_filename):
    img = read_img(in_filename, swap_rb=swap_rb, signed=True, noise=noise)

    if scale:
        img = skimage.transform.rescale(img, scale, channel_axis=2)

    if run_size:
        original_shape = img.shape[:2]
        run_scale = max(run_size / img.shape[0], run_size / img.shape[1])
        print("run_scale", run_scale)
        img = skimage.transform.rescale(img, run_scale, channel_axis=2)

    out_img = run_img(preview_session, upscale_session, img)

    if run_size:
        out_img = skimage.transform.resize(out_img, original_shape)

    write_img(
        out_filename, out_img, swap_rb=swap_rb, signed=True, output_8_bit=output_8_bit
    )


if __name__ == "__main__":
    preview_session = create_session(preview_model_filename)
    upscale_session = create_session(upscale_model_filename)

    out_extname = ".png" if (output_png or not output_8_bit) else None
    for in_pattern in in_filenames:
        for in_filename in glob.glob(in_pattern):
            print(in_filename)
            basename, extname = os.path.splitext(in_filename)
            out_filename = basename + (out_suffix or "") + (out_extname or extname)
            convert_img(preview_session, upscale_session, in_filename, out_filename)
