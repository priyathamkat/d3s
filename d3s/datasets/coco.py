from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from d3s.constants import COCO_ROOT
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import Lambda

MASK_TRANSFORM = Lambda(lambda x: torch.from_numpy(x))


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = MASK_TRANSFORM,
        transforms: Optional[Callable] = None,
    ) -> None:
        root = COCO_ROOT / f"images/{split}2017"
        annFile = COCO_ROOT / f"annotations/instances_{split}2017.json"
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.classes = [
            cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self._cat = None
        self.mask_transform = mask_transform
        self._rng = np.random.default_rng()

    @property
    def cat(self):
        return self._cat

    @cat.setter
    def cat(self, c):
        self._cat = c
        if c == "all":
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.catId = None
        else:
            self.catId = self.coco.getCatIds(catNms=[c])[0]
            self.ids = sorted(self.coco.getImgIds(catIds=[self.catId]))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(self.root / path).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _get_consolidated_mask(self, target) -> Any:
        consolidated_mask = None
        for segmentation in target:
            if segmentation["category_id"] == self.catId:
                mask = self.coco.annToMask(segmentation)
                try:
                    consolidated_mask += mask
                except TypeError:
                    consolidated_mask = mask

        return np.expand_dims(consolidated_mask, -1)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.catId is not None:
            mask = self._get_consolidated_mask(target)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, target, mask
        else:
            return image, target

    def get_random(self, _):
        index = self._rng.choice(len(self))
        return self[index]

    def __len__(self) -> int:
        return len(self.ids)
