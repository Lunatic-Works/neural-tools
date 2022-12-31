import numpy as np
import PIL
from PIL import Image, ImageDraw

_image_draw = ImageDraw.Draw(Image.new("L", (1, 1), 255))


# Outputs in [-1, 1]
def pil_to_np(img):
    return np.asarray(img, dtype=np.float32) / 127.5 - 1


def char_to_np(ch, font, canvas_size, char_size):
    width, height = _image_draw.textsize(ch, font=font)

    bg = Image.new("L", (canvas_size, canvas_size), 255)
    if width == 0 or height == 0:
        return pil_to_np(bg)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, fill=0, font=font)

    factor = width / char_size
    max_height = canvas_size * 2
    if height / factor > max_height:
        # Too long
        img = img.crop((0, 0, width, int(max_height * factor)))

    if height / factor > char_size + 5:
        factor = height / char_size

    img = img.resize(
        (int(width / factor), int(height / factor)), resample=PIL.Image.LANCZOS
    )

    offset = ((canvas_size - img.size[0]) // 2, (canvas_size - img.size[1]) // 2)
    bg.paste(img, offset)

    return pil_to_np(bg)


def text_to_np(text, font, canvas_size, char_size):
    data = [char_to_np(ch, font, canvas_size, char_size) for ch in text]
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
    img.save(filename)
