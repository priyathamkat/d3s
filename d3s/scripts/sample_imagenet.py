from pathlib import Path

import torchvision.transforms as T
from absl import app, flags
from d3s.datasets import ImageNet
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("save_path", None, "Path to save sample images")
flags.DEFINE_integer("num_per_class", 4, "Number of images to save per class")


def main(argv):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
    imagenet = ImageNet(transform=transform)
    save_path = Path(FLAGS.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    for class_idx in tqdm(imagenet.classes):
        for i in range(FLAGS.num_per_class):
            image, _ = imagenet.get_random(class_idx)
            image.save(save_path / f"{class_idx}_{i}.jpg")


if __name__ == "__main__":
    flags.mark_flags_as_required(["save_path"])
    app.run(main)
