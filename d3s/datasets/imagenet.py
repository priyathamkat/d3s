import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torchvision.datasets as D
from d3s.constants import IMAGENET_PATH


class ImageNet(D.ImageNet):
    def __init__(self, split: str = "train", **kwargs: Any) -> None:
        super().__init__(IMAGENET_PATH, split, **kwargs)

        with open(Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r") as f:
            self.classes = {int(k): v.split(",")[0] for k, v in json.load(f).items()}

        with open(Path(__file__).parent.parent / "metadata/imagenet_dictionary.json", "r") as f:
            self.dictionary = {int(k): v for k, v in json.load(f).items()}

        self.class_to_indices = defaultdict(list)
        for idx, target in enumerate(self.targets):
            self.class_to_indices[target].append(idx)

        self._rng = np.random.default_rng()

    def get_random(self, class_idx: int, num_samples: int = 1) -> Any:
        options = self.class_to_indices[class_idx]
        return [self[idx] for idx in self._rng.choice(options, size=num_samples)]
