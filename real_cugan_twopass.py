#!/usr/bin/env python3

import glob
import os

import cv2
import numpy as np
import skimage.transform

from utils import (
    create_session,
    floor_even,
    get_batch,
    get_tiles,
    merge_img,
    read_img,
    trim_img,
    untrim_img,
    write_img,
)

preview_model_filename = "./models/real_cugan_twopass/up2x_latest_denoise3x_preview.onnx"
upscale_model_filename = "./models/real_cugan_twopass/up2x_latest_denoise3x_upscale.onnx"
in_filenames = [
    "./in.png",
]
out_suffix = "_denoise3x"

preview_size = 512
preview_min_side = 64

tile_inner_size = 176
pad_size = 20
up_scale = 2
up_shift = 0
batch_size = 16

# pre_blur = 0
pre_blur = 0.5
pre_darken = False
pre_lighten = False
# pre_blur = 1
# pre_darken = True
# pre_lighten = True
trim_alpha = False
trim_eps = 1e-3
upscale = True
downscale = True
# upscale = False
# downscale = False
run_alpha = False
alpha_gamma = 1
wrap_x = False
wrap_y = False
output_gray = False
output_alpha = False
output_png = True
output_8_bit = True

if trim_alpha or run_alpha:
    assert output_alpha


def compute_means(preview_session, img):
    h, w = img.shape[:2]
    scale = min(1.0, preview_size / max(h, w))
    ph = max(preview_min_side, floor_even(int(round(h * scale))))
    pw = max(preview_min_side, floor_even(int(round(w * scale))))
    if (ph, pw) != (h, w):
        small = skimage.transform.resize(img, (ph, pw))
    else:
        small = img
    x = np.ascontiguousarray(small.transpose(2, 0, 1)[None]).astype(np.float32)
    return preview_session.run(None, {"in": x})


def run_img(preview_session, upscale_session, img):
    means = compute_means(preview_session, img)

    tiles, max_row_col, pads = get_tiles(
        img, tile_inner_size, pad_size, wrap_x=wrap_x, wrap_y=wrap_y
    )

    out_tiles = []
    for batch in get_batch(tiles, batch_size):
        feed = {"in": batch}
        for i, mean in enumerate(means):
            feed[f"mean{i}"] = mean
        out_batch = upscale_session.run(None, feed)[0]
        out_batch = out_batch.transpose(0, 2, 3, 1)
        out_tiles.append(out_batch)
    out_tiles = np.concatenate(out_tiles)

    out_img = merge_img(
        out_tiles, tile_inner_size, pad_size, max_row_col, pads, (up_scale, up_shift)
    )
    return out_img


def convert_img(preview_session, upscale_session, in_filename, out_filename):
    img, alpha = read_img(in_filename, swap_rb=True, signed=False, return_alpha=True)

    if pre_blur:
        img_blur = cv2.GaussianBlur(img, (0, 0), pre_blur)
    else:
        img_blur = img

    if pre_darken:
        assert pre_blur
        assert not pre_lighten
        img = np.minimum(img, img_blur)
    elif pre_lighten:
        assert pre_blur
        img = np.maximum(img, img_blur)
    else:
        img = img_blur
    del img_blur

    if trim_alpha and alpha is not None:
        original_shape, (trim_t, trim_b, trim_l, trim_r) = trim_img(
            img, alpha, trim_eps
        )

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
                alpha = skimage.transform.resize(alpha, floor_even(alpha.shape))
            else:
                alpha = skimage.transform.resize(
                    alpha, (alpha.shape[0] * 2, alpha.shape[1] * 2)
                )
    else:
        img = skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
        if alpha is not None:
            if run_alpha:
                alpha = skimage.transform.resize(
                    alpha, (alpha.shape[0] // 2, alpha.shape[1] // 2)
                )
            else:
                alpha = skimage.transform.resize(alpha, floor_even(alpha.shape))

    img = run_img(preview_session, upscale_session, img)

    if run_alpha:
        alpha = np.repeat(alpha[:, :, None], 3, axis=2)
        alpha = run_img(preview_session, upscale_session, alpha)
        alpha = alpha.mean(axis=2)
        alpha = np.clip(alpha, 0, 1)
        alpha[alpha < 1 - trim_eps] **= alpha_gamma

    if upscale and downscale:
        img = skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
        if run_alpha:
            alpha = skimage.transform.resize(
                alpha, (alpha.shape[0] // 2, alpha.shape[1] // 2)
            )

    if trim_alpha and alpha is not None:
        img, alpha = untrim_img(
            img, alpha, original_shape, (trim_t, trim_b, trim_l, trim_r)
        )

    write_img(
        out_filename,
        img,
        alpha=alpha if output_alpha else None,
        swap_rb=True,
        signed=False,
        output_gray=output_gray,
        output_8_bit=output_8_bit,
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
