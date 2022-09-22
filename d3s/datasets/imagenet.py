from typing import Any

import torchvision.datasets as D


class ImageNet(D.ImageNet):
    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)

        with open("./txt_data/imagenet_classes.txt", "r") as f:
            self.classes = eval(f.read())
