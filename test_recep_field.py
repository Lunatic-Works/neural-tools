#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import onnxruntime as rt
import skimage.transform

from utils import get_tiles, merge_img, read_img


def load_input(image, size, *, signed, noise, swap_rb, seed=0):
    if image and os.path.exists(image):
        img = read_img(image, swap_rb=swap_rb, signed=signed, noise=noise)
        h, w = img.shape[:2]
        if h < size or w < size:
            scale = max(size / h, size / w)
            img = skimage.transform.rescale(img, scale, channel_axis=2)
            h, w = img.shape[:2]
        t, l = (h - size) // 2, (w - size) // 2
        return np.ascontiguousarray(img[t : t + size, l : l + size]).astype(np.float32)

    rng = np.random.default_rng(seed)
    low = np.kron(
        rng.random((16, 16, 3)).astype(np.float32),
        np.ones((size // 16 + 1, size // 16 + 1, 1), np.float32),
    )[:size, :size]
    img = 0.7 * low + 0.3 * rng.random((size, size, 3), dtype=np.float32)
    if signed:
        img = img * 2 - 1
    if noise:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.ascontiguousarray(img).astype(np.float32)


def make_session(model_filename):
    so = rt.SessionOptions()
    so.log_severity_level = 4  # silence cuDNN frontend chatter
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    po = [{"cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "0"}, {}]
    return rt.InferenceSession(
        model_filename, so, providers=providers, provider_options=po
    )


def run_full(sess, x):
    y = sess.run(None, {"in": np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0]
    return y[0].transpose(1, 2, 0)


def run_tiled(sess, x, tile, pad, scale, shift, batch=4):
    tiles, max_row_col, pads = get_tiles(x, tile, pad)
    outs = []
    for b in range(0, len(tiles), batch):
        t = np.ascontiguousarray(tiles[b : b + batch].transpose(0, 3, 1, 2))
        outs.append(sess.run(None, {"in": t})[0].transpose(0, 2, 3, 1))
    return merge_img(np.concatenate(outs), tile, pad, max_row_col, pads, (scale, shift))


def sweep(model_filename, tile, pads, *, args):
    size = args.size or 3 * tile
    x = load_input(
        args.image, size, signed=args.signed, noise=args.noise, swap_rb=args.swap_rb
    )
    sess = make_session(model_filename)
    y_full = run_full(sess, x)

    margin = args.margin or tile
    lo, hi = margin * args.scale, y_full.shape[0] - margin * args.scale
    lv = 127.5 if args.signed else 255.0
    rows = []
    for pad in pads:
        y_t = run_tiled(sess, x, tile, pad, args.scale, args.shift)
        d = np.abs(y_full[lo:hi, lo:hi] - y_t[lo:hi, lo:hi])
        e = d * lv
        rows.append(
            (pad, float(e.max()), float(e.mean()), float(np.percentile(e, 99.9)))
        )
        print(
            f"pad={pad:4d} "
            f"max={rows[-1][1]:8.4f} "
            f"mean={rows[-1][2]:8.4f} "
            f"p99.9={rows[-1][3]:8.4f} lv"
        )
    return rows


def first_pad(rows, threshold):
    for pad, mx, _, _ in rows:
        if mx <= threshold:
            return pad
    return None


def gradient_extent(model_filename, args):
    sess = make_session(model_filename)
    S = args.size or 256
    i0 = j0 = S // 2
    dx, eps = 1e-3, 1e-7
    g_sqr = 0.0
    rng = np.random.default_rng(0)
    for _ in range(args.batch_count):
        x0 = rng.random((1, 3, S, S), dtype=np.float32)
        x1 = x0.copy()
        x1[:, :, i0, j0] += dx
        y = sess.run(None, {"in": np.concatenate([x0, x1], axis=0)})[0]
        g_sqr += ((y[1] - y[0]) / dx) ** 2
    g = np.sqrt(g_sqr / args.batch_count).mean(axis=0)
    i = i0
    while i < g.shape[0] and abs(g[i, j0]) >= eps:
        i += 1
    j = j0
    while j < g.shape[1] and abs(g[i0, j]) >= eps:
        j += 1
    print(f"Gradient extent (eps={eps}): down={i - i0}, right={j - j0}")
    return i - i0, j - j0


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["tiled", "gradient"], default="tiled")
    parser.add_argument("--tile", type=int, help="inner tile size")
    parser.add_argument("--pads", type=int, nargs="*", help="pad sizes to test")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--signed", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--swap-rb", action="store_true")
    parser.add_argument("--image", help="input image (default: synthetic pattern)")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--margin", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--batch-count", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "gradient":
        gradient_extent(args.model, args)
        return

    if args.tile is None or not args.pads:
        parser.error("--tile and --pads are required in tiled mode")
    rows = sweep(args.model, args.tile, args.pads, args=args)
    for thr in (1.0, 0.5, args.threshold):
        p = first_pad(rows, thr)
        print(f"Smallest pad with max error <= {thr:g} lv: {p}")
    if min(r[1] for r in rows) > 1.0:
        print(
            "WARNING: no pad converges. "
            "The model is not tile-consistent (not a receptive-field issue)"
        )


if __name__ == "__main__":
    sys.exit(main())
