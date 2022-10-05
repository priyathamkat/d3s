import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

from PIL import Image
from torch.utils.data import Dataset


class D3S(Dataset):
    def __init__(
        self, root: Union[str, Path], transform=None, target_transform=None
    ) -> None:
        super().__init__()
        self.root = Path(root)
        with open(self.root / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

        self.images = {}
        self.class_to_indices = defaultdict(list)
        for k, v in self.metadata.items():
            idx = int(k[: k.rfind(".")])
            class_idx = int(v["args"]["class_idx"])
            self.class_to_indices[class_idx].append(idx)
            self.images[idx] = {
                "image_path": self.root / k,
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
        return len(self.metadata)


if __name__ == "__main__":
    dataset = D3S("/cmlscratch/pkattaki/datasets/d3s/pd")
    print(dataset.class_to_indices)
