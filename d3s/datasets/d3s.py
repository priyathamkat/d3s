import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

from PIL import Image
from torch.utils.data import Dataset


class D3S(Dataset):
    shifts = ["all", "background-shift", "geography-shift", "time-shift"]

    def __init__(
        self, root: Union[str, Path], shift="all", transform=None, target_transform=None
    ) -> None:
        super().__init__()
        self.root = Path(root)
        with open(self.root / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

        assert shift in self.shifts, f"shift must be one of {self.shifts}"

        self.images = {}
        self.class_to_indices = defaultdict(list)
        for idx, metadatum in enumerate(self.metadata):
            image_path = metadatum["image"]
            if shift != "all" and shift not in image_path:
                continue
            class_idx = metadatum["classIdx"]
            self.class_to_indices[class_idx].append(idx)
            self.images[idx] = {
                "image_path": image_path,
                "class_idx": class_idx,
            }

    def __getitem__(self, idx: Any) -> Any:
        image = self.images[idx]["image_path"]
        image = Image.open(image)

        label = self.images[idx]["class_idx"]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    dataset = D3S("/cmlscratch/pkattaki/datasets/d3s/", shift="background-shift")
    print(dataset.images[0])
    print(len(dataset))
