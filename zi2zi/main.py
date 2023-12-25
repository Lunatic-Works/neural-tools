#!/use/bin/env python3

import onnxruntime as rt
from PIL import ImageFont

from utils import save_text, text_to_np

model_filename = "../models/zi2zi/qingxiang.onnx"
src_font_filename = "./SimSun.ttf"
text = "海滨小城里，缓缓流动的时光中，普通人的故事"
out_dir = "./out/"

canvas_size = 256
char_size = 220
dst_embedding_id = 38


def main():
    sess = rt.InferenceSession(model_filename)

    src_font = ImageFont.truetype(src_font_filename, size=char_size)
    real_data = text_to_np(text, src_font, canvas_size)
    embedding_ids = [dst_embedding_id] * real_data.shape[0]

    fake_data = sess.run(
        ["generator_1/Tanh:0"],
        {"real_A_and_B_images:0": real_data, "embedding_ids:0": embedding_ids},
    )[0]

    out_filename = f"{out_dir}/font_{dst_embedding_id}/out_{text}.png"
    print(out_filename)
    save_text(out_filename, fake_data)


if __name__ == "__main__":
    main()
