import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

from PIL import Image
from torch.utils.data import Dataset


class D3S(Dataset):
    shifts = ["all", "background-shift", "geography-shift", "time-shift"]

    def __init__(
        self, root: Union[str, Path], shift="all", return_init=False, transform=None, target_transform=None
    ) -> None:
        super().__init__()
        self.root = Path(root)
        
        with open(Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r") as f:
            self.classes = {int(k): v.split(",")[0] for k, v in json.load(f).items()}
        
        with open(self.root / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.return_init = return_init
        self.transform = transform
        self.target_transform = target_transform

        assert shift in self.shifts, f"shift must be one of {self.shifts}"

        self.images = {}
        self.class_to_indices = defaultdict(list)
        idx = 0
        for metadatum in self.metadata:
            image_path = metadatum["image"]
            if shift != "all" and shift not in image_path:
                continue
            class_idx = metadatum["classIdx"]
            self.class_to_indices[class_idx].append(idx)
            self.images[idx] = {
                "image_path": image_path,
                "class_idx": class_idx,
            }
            idx += 1

    def __getitem__(self, idx: Any) -> Any:
        image = self.images[idx]["image_path"]
        image = Image.open(image)

        if not self.return_init:
            image = image.crop((518, 2, 518 + 512, 2 + 512))
        label = int(self.images[idx]["class_idx"])

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
    print(dataset[0][0].size)
    print(len(dataset))
