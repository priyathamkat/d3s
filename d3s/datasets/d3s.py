import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class D3S(Dataset):
    shifts = ["all", "background-shift", "geography-shift", "time-shift"]

    def __init__(
        self,
        root: Union[str, Path],
        split="train",
        shift="all",
        return_init=False,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        self.split = split
        assert self.split in ["train", "val"]
        self.root = Path(root) / self.split

        with open(
            Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r"
        ) as f:
            self.classes = {int(k): v.split(",")[0] for k, v in json.load(f).items()}

        with open(self.root / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.return_init = return_init
        self.transform = transform
        self.target_transform = target_transform

        assert shift in self.shifts, f"shift must be one of {self.shifts}"

        self.images = {}
        self.class_to_indices = defaultdict(list)
        self.backgrounds = None
        self.backgrounds_to_indices = None
        if shift == "background-shift":
            with open(
                Path(__file__).parent.parent / "recipes/prompts/options.yaml", "r"
            ) as f:
                options = yaml.load(f, Loader=yaml.CLoader)
            self.backgrounds = [s.split(" ")[-1] for s in list(options[0].values())[0]]
            self.backgrounds_to_indices = defaultdict(list)

        idx = 0
        for metadatum in self.metadata:
            image_path = metadatum["image"]
            if shift != "all" and shift not in image_path:
                continue
            class_idx = int(metadatum["classIdx"])

            self.class_to_indices[class_idx].append(idx)
            self.images[idx] = {
                "image_path": image_path,
                "class_idx": class_idx,
            }
            if self.backgrounds:
                background = metadatum["background"]
                bg_idx = self.backgrounds.index(background)
                self.images[idx]["bg_idx"] = bg_idx
                self.backgrounds_to_indices[bg_idx].append(idx)
            idx += 1

        self._rng = np.random.default_rng()

    def __getitem__(self, idx: Any) -> Any:
        image = self.images[idx]["image_path"]
        image = Image.open(image)

        if not self.return_init:
            image = image.crop((518, 2, 518 + 512, 2 + 512))
        label = self.images[idx]["class_idx"]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.backgrounds:
            bg_idx = self.images[idx]["bg_idx"]
            return image, label, bg_idx
        else:
            return image, label

    def get_random(
        self,
        class_idx: Optional[int] = None,
        bg_idx: Optional[int] = None,
        not_class_idx: Optional[int] = None,
        not_bg_idx: Optional[int] = None,
        num_samples: int = 1,
    ) -> Any:
        options = set(range(len(self)))
        if class_idx:
            options.intersection_update(set(self.class_to_indices[class_idx]))
        if bg_idx:
            options.intersection_update(set(self.backgrounds_to_indices[bg_idx]))
        if not_class_idx:
            options.difference_update(set(self.class_to_indices[not_class_idx]))
        if not_bg_idx:
            options.difference_update(set(self.backgrounds_to_indices[not_bg_idx]))

        options = list(options)
        return [self[idx] for idx in self._rng.choice(options, size=num_samples)]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    dataset = D3S(
        "/cmlscratch/pkattaki/datasets/d3s/", split="train", shift="background-shift"
    )
    print(dataset.backgrounds)
    print(dataset.images[0])
    print(dataset[0][0].size)
    print(len(dataset))
