from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from d3s.datasets import ImageNet
from d3s.constants import IMAGENET_C_PATH


class CorruptionDataset(Dataset):

    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "all",
    ]

    severities = ["1", "2", "3", "4", "5", "all"]

    def __init__(self, corruption, severity):
        super().__init__()

        assert (
            corruption in CorruptionDataset.corruptions
        ), f"Unknown corruption: {corruption}"
        assert severity in CorruptionDataset.severities, f"Unknown severity: {severity}"

        self.corruption_idx = CorruptionDataset.corruptions.index(corruption)


class ImageNetC(CorruptionDataset):
    def __init__(
        self,
        corruption,
        severity,
        return_labels=True,
        return_corruption=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(corruption, severity)

        self._rng = np.random.default_rng()

        self.imagenetc_root = IMAGENET_C_PATH

        self.corruption = corruption
        self.severity = severity
        self.return_labels = return_labels

        self._imagenet = ImageNet(split="val")
        self.classes = self._imagenet.classes
        self.dictionary = self._imagenet.dictionary
        self.class_to_indices = self._imagenet.class_to_indices

        self.return_corruption = return_corruption
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        clean_image, label = self._imagenet.samples[idx]
        clean_image = Path(clean_image)

        if self.corruption == "all":
            corruption = self._rng.choice(self.corruptions[:-1])
        else:
            corruption = self.corruption

        if self.severity == "all":
            severity = self._rng.choice(self.severities[:-1])
        else:
            severity = self.severity

        corruption_image = (
            self.imagenetc_root
            / corruption
            / severity
            / clean_image.parent.name
            / clean_image.name
        )
        corruption_image = Image.open(corruption_image)
        if corruption_image.mode != "RGB":
            corruption_image = corruption_image.convert("RGB")

        if self.transform:
            corruption_image = self.transform(corruption_image)
        if self.return_labels and self.target_transform:
            label = self.target_transform(label)

        item = [corruption_image]
        if self.return_labels:
            item.append(label)
        if self.return_corruption:
            item.append(self.corruption_idx)

        return tuple(item)

    def __len__(self):
        return len(self._imagenet.samples)

    def get_random(self, class_idx: int, num_samples: int = 1):
        options = self.class_to_indices[class_idx]
        return [self[idx] for idx in self._rng.choice(options, size=num_samples)]
