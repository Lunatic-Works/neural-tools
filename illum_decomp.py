#!/usr/bin/env python3

import numpy as np

from utils import read_img, write_img

in_filename = r"./in.png"
base_filename = in_filename.replace(".png", "_simplify_out0.png")
shadow_filename = in_filename.replace(".png", "_shadow.png")
light_filename = in_filename.replace(".png", "_light.png")
eps = 1e-7


def main():
    img = read_img(in_filename, swap_rb=True, signed=False)
    base = read_img(base_filename, swap_rb=True, signed=False)

    shadow = np.clip(img / np.maximum(base, eps), 0, 1)
    light = np.clip(1 - (1 - img) / np.maximum(1 - base, eps), 0, 1)

    write_img(shadow_filename, shadow, swap_rb=True, signed=False, output_8_bit=False)
    write_img(light_filename, light, swap_rb=True, signed=False, output_8_bit=False)


if __name__ == "__main__":
    main()
