import numpy as np
from PIL import Image

color_replace_map = {  # ( blue, green, red)
    0: (0, 0, 0),
    1: (0,0,255),
    2: ( 0, 255, 0),
    3: ( 255,0,0),
    4: ( 0, 255,255),
}

color_replace_map2 = {  # RGB
    (0, 0, 0):0,
    (255,0,0):64,
    ( 0, 255, 0):128,
    ( 0,0,255):192,
    ( 255, 255,0):255,
}


def pic_replace_color(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=2)

    out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for key, val in color_replace_map.items():
        i, j, k = np.where(img == key)
        out[i, j, :] = val
    return out

def replace_color_3to1(img):
    img_array = np.array(img)

    output_array = np.zeros(img_array.shape[:2], dtype=np.uint8)

    for original_color, new_color in color_replace_map2.items():
        matches = (img_array == original_color).all(axis=-1)

        output_array[matches] = new_color

    output_img = Image.fromarray(output_array, mode='L')
    return output_img
