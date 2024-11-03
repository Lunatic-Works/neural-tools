#!/usr/bin/env python3

import numpy as np
import onnxruntime as rt
from tqdm import tqdm

model_filename = "./models/cartoonize/shinkai.onnx"
# model_filename = "./models/cartoonize_2/danbooru.onnx"
# model_filename = "./models/real_cugan/up2x_latest_denoise3x.onnx"
# model_filename = "./models/stylize/udnie.onnx"
# model_filename = "./models/stylize_2/van_gogh.onnx"
# model_filename = "./models/waifu2x/noise0_scale2x.onnx"
size = 256
batch_count = 10
batch_size = 1
n_channels = 3
scale = 1
shift = 0
dx = 1e-3
eps = 1e-7


def main():
    sess = rt.InferenceSession(
        model_filename,
        providers=[
            # "TensorrtExecutionProvider",
            # "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    i0 = size // 2
    j0 = size // 2
    rng = np.random.default_rng()
    g_sqr_sum = 0
    for _ in tqdm(range(batch_count)):
        x0 = rng.random((batch_size, n_channels, size, size), dtype=np.float32)
        x1 = x0.copy()
        x1[:, :, i0, j0] += dx
        x = np.concatenate([x0, x1], axis=0)

        y = sess.run(["out"], {"in": x})[0]

        y0 = y[:batch_size]
        y1 = y[batch_size:]
        g = (y1 - y0) / dx
        g_sqr_sum += (g**2).sum(axis=1).mean(axis=0)

    g = np.sqrt(g_sqr_sum / batch_count)
    i0 = i0 * scale - shift
    j0 = j0 * scale - shift
    print("g", g.shape, g.dtype, i0, j0)

    for i in range(i0, g.shape[0]):
        if abs(g[i, j0]) < eps:
            break
    for j in range(j0, g.shape[1]):
        if abs(g[i0, j]) < eps:
            break
    print(i - i0, j - j0)

    # import imageio
    # import skimage
    # img = np.stack([g, g, g], axis=2)
    # img = skimage.img_as_ubyte(img / img.max())
    # imageio.imsave('./g.png', img)


if __name__ == "__main__":
    main()
