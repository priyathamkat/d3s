from pathlib import Path
from typing import Any

import torchvision.datasets as D
from d3s.constants import IMAGENET_PATH


class ImageNet(D.ImageNet):
    def __init__(self, split: str = "train", **kwargs: Any) -> None:
        super().__init__(IMAGENET_PATH, split, **kwargs)

        with open(Path(__file__).parent.parent / "txt_data/imagenet_classes.txt", "r") as f:
            self.classes = {int(k): v.split(",")[0] for k, v in eval(f.read()).items()}

        with open(Path(__file__).parent.parent / "txt_data/imagenet_dictionary.txt", "r") as f:
            self.dictionary = {int(k): v for k, v in eval(f.read()).items()}
