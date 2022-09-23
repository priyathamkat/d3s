from typing import Any

import torchvision.datasets as D
from d3s.constants import IMAGENET_PATH


class ImageNet(D.ImageNet):
    def __init__(self, split: str = "train", **kwargs: Any) -> None:
        super().__init__(IMAGENET_PATH, split, **kwargs)

        with open("./txt_data/imagenet_classes.txt", "r") as f:
            self.classes = eval(f.read())

        with open("./txt_data/imagenet_dictionary.txt", "r") as f:
            self.dictionary = eval(f.read())
