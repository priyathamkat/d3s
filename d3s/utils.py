import numpy as np
import torchvision.transforms as T
from PIL import Image

rng = np.random.default_rng()


def resize(x, size):
    return T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)(x)


def pad_to_squarize(x):
    w, h = x.shape[:2]
    abs_diff = abs(w - h)
    before_pad = abs_diff // 2
    after_pad = abs_diff - before_pad
    pad_width = [(0, 0), (before_pad, after_pad)]
    if w < h:
        pad_width.reverse()
    if x.ndim > 2:
        pad_width.append((0, 0))
    return np.pad(x, pad_width)


def crop_to_squarize(x):
    w, h = x.size
    return T.CenterCrop(min(w, h))(x)


def paste_on_bg(fg, bg, mask=None, size=512, fg_scale=0.4):
    if mask is None:
        bg = crop_to_squarize(bg)
        bg = resize(bg, size)

        fg = crop_to_squarize(fg)
        fg = resize(fg, int(fg_scale * size))

        image = np.array(bg)
        fg = np.array(fg)

        w, h = fg.shape[:2]

        w_start = rng.choice(image.shape[0] - w)
        h_start = rng.choice(image.shape[1] - h)

        image[w_start : w_start + w, h_start : h_start + h, :] = fg

        return Image.fromarray(image)
    else:
        mask = pad_to_squarize(mask)
        fg = pad_to_squarize(np.array(fg))

        bg = crop_to_squarize(bg)
        bg = resize(bg, fg.shape[0])
        bg = np.array(bg)

        image = Image.fromarray(mask * fg + (1 - mask) * bg)
        image = resize(image, size)
        return image
