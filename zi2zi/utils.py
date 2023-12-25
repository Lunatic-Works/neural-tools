import os

import numpy as np
from PIL import Image, ImageDraw


# Outputs in [-1, 1]
def pil_to_np(img):
    return np.asarray(img, dtype=np.float32) / 127.5 - 1


def char_to_np(ch, font, canvas_size):
    _, _, width, height = font.getbbox(ch)

    bg = Image.new("L", (canvas_size, canvas_size), 255)
    if width == 0 or height == 0:
        return pil_to_np(bg)

    offset = ((canvas_size - width) // 2, (canvas_size - height) // 2)
    draw = ImageDraw.Draw(bg)
    draw.text(offset, ch, fill=0, font=font)
    return pil_to_np(bg)


def text_to_np(text, font, canvas_size):
    data = [char_to_np(ch, font, canvas_size) for ch in text]
    data = np.stack(data, axis=0)
    data = np.stack([data, data], axis=3)
    return data


def save_text(filename, data, vertical=False):
    canvas_size = data.shape[1]

    data = np.clip(data, -1, 1)
    data = (data + 1) * 127.5
    data = data.astype(np.uint8)
    data = data.squeeze(axis=-1)

    if vertical:
        data = data.reshape((-1, canvas_size))
    else:
        data = data.transpose((1, 0, 2)).reshape((canvas_size, -1))

    img = Image.fromarray(data, "L")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)
