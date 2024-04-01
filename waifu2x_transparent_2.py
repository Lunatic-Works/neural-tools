#!/usr/bin/env python3

import cv2
import numpy as np
import skimage.transform

from utils import do_imgs, floor_even, read_img, trim_img, untrim_img, write_img
from waifu2x import pad_size, run_img

model_filename = "./models/waifu2x/noise0_scale2x.onnx"
in_filenames = [
    "./in.png",
]
out_suffix = "_waifu2x"

pre_blur = 0.5
pre_darken = False
pre_lighten = False
trim_eps = 1e-3

waifu2x_strength = 0.7
waifu2x_alpha = False
alpha_blur_scale = 0.7
alpha_blur_gamma = 1
alpha_blur_strength = 0.7

use_gpu = True

if use_gpu:
    import cupy as cp
else:
    cp = np
    cp.asnumpy = lambda x: x


def cross_sum(a, out):
    a = cp.pad(a, ((1, 1), (1, 1), (0, 0)))
    out[:] = a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]


def bleed_alpha(img, alpha, eps, max_iter=10**3, tol=1e-4):
    img = cp.asarray(img)
    alpha = cp.asarray(alpha)

    assert alpha.ndim == 2
    alpha = alpha[:, :, None]
    mask_0 = alpha > eps
    mask = mask_0.astype(img.dtype)
    # TODO: np.maximum breaks type stability in numba
    confidence = cp.maximum(alpha, eps).astype(img.dtype)
    del alpha

    img_new = cp.zeros_like(img)
    for iter_count in range(max_iter):
        cross_sum(mask * confidence * img, img_new)
        cross_sum(mask * confidence, mask)
        img_new /= cp.maximum(mask, 1e-7).astype(mask.dtype)
        img_new = cp.where(mask_0, img, img_new)

        norm = ((img_new - img) ** 2).max()
        if norm < tol:
            break

        img, img_new = img_new, img
        mask = (mask > 0).astype(mask.dtype)

    img_new = cp.asnumpy(img_new)
    return iter_count + 1, img_new


def do_blur(img):
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

    return img


def convert_img(sess, in_filename, out_filename):
    # Network input is BGR
    img, alpha = read_img(in_filename, swap_rb=False, signed=False, return_alpha=True)

    original_shape, (trim_t, trim_b, trim_l, trim_r) = trim_img(
        img, alpha, trim_eps, pad=pad_size
    )
    trim_b = trim_t + floor_even(trim_b - trim_t)
    trim_r = trim_l + floor_even(trim_r - trim_l)
    img = img[trim_t:trim_b, trim_l:trim_r, :]
    alpha = alpha[trim_t:trim_b, trim_l:trim_r]

    iter_count, img = bleed_alpha(img, alpha, trim_eps)
    print("Bleed alpha iter", iter_count)

    img_out = do_blur(img)

    img_out = skimage.transform.resize(img_out, (img.shape[0] // 2, img.shape[1] // 2))
    img_out = run_img(sess, img_out)
    img = (1 - waifu2x_strength) * img + waifu2x_strength * img_out
    del img_out

    alpha_blur = do_blur(alpha)

    if waifu2x_alpha:
        alpha_blur = skimage.transform.resize(
            alpha_blur, (alpha.shape[0] // 2, alpha.shape[1] // 2)
        )
        alpha_blur = np.repeat(alpha_blur[:, :, None], 3, axis=2)
        alpha_blur = run_img(sess, alpha_blur)
        alpha_blur = alpha_blur.mean(axis=2)
    else:
        alpha_blur = skimage.transform.resize(
            alpha_blur, tuple(int(alpha_blur_scale * x) for x in alpha.shape)
        )
        alpha_blur = skimage.transform.resize(alpha_blur, alpha.shape)
    alpha_blur **= alpha_blur_gamma
    alpha = (1 - alpha_blur_strength) * alpha + alpha_blur_strength * alpha_blur
    del alpha_blur

    img, alpha = untrim_img(
        img, alpha, original_shape, (trim_t, trim_b, trim_l, trim_r)
    )

    # Network output is BGR
    write_img(out_filename, img, alpha, swap_rb=False, signed=False)


if __name__ == "__main__":
    do_imgs(convert_img, model_filename, in_filenames, out_suffix)
