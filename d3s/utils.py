from typing import Any, List

import numpy as np
import torchvision.transforms as T
from IPython.display import clear_output, display
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


class ScoreTracker:
    def __init__(self, name: str, low: int = 1, high: int = 5) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.score = 0
        self.n = 1
        self._scores = []

    def update(self, value):
        assert self.low <= value <= self.high
        self._scores.append(value)
        self.score += (value - self.score) / self.n
        self.n += 1

    def __str__(self) -> str:
        percentage = 100 * (self.score - self.low) / (self.high - self.low)
        return f"Average {self.name} ({self.low} to {self.high}) over {self.n - 1} samples: {self.score} ({percentage:.2f})"


def format_attr_name(name):
    return name.lower().replace(" ", "_")


class ExperimentTracker:
    def __init__(self, name: str, metrics: List[Any]) -> None:
        self.name = name
        self.metrics = []
        for metric in metrics:
            if isinstance(metric, tuple):
                metric, low, high = metric
                metric_attr_name = format_attr_name(metric)
                setattr(
                    self, metric_attr_name, ScoreTracker(metric, low=low, high=high)
                )
            else:
                metric_attr_name = format_attr_name(metric)
                setattr(self, metric_attr_name, ScoreTracker(metric))

            self.metrics.append(getattr(self, metric_attr_name))

    def __str__(self) -> str:
        output = f"{self.name} experiment\n"
        for metric in self.metrics:
            output += f"{str(metric)}\n"
        return output


def experiment(name: str, images: List[Any], metrics: List[Any]):
    tracker = ExperimentTracker(name, metrics)
    for prompt, image in images:
        clear_output(wait=True)
        print(f"Class: {prompt}")
        display(image)
        for metric in tracker.metrics:
            try:
                value = int(
                    input(f"Score for {metric.name} ({metric.low} to {metric.high}):")
                )
            except AssertionError:
                print("Try again")
                value = int(
                    input(f"Score for {metric.name} ({metric.low} to {metric.high}):")
                )
            metric.update(value)
    print("-----Results-----")
    print(tracker)
