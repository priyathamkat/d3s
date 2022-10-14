from pathlib import Path

import numpy as np
from absl import app, flags
from d3s.datasets import D3S
from PIL import ImageDraw, ImageFont
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import trange

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_enum(
    "shift",
    "all",
    ["all", "background-shift", "geography-shift", "time-shift"],
    "Shift to use for evaluation",
)
flags.DEFINE_integer("num_images", 49, "Number of images to show.")
flags.DEFINE_string("output_path", None, "Path of the output image.")
flags.DEFINE_bool(
    "include_class_names",
    True,
    "Whether to include the class names in the output image.",
)


def main(argv):
    rng = np.random.default_rng(0)
    dataset = D3S(Path(FLAGS.d3s_root), return_init=True, shift=FLAGS.shift)
    assert FLAGS.num_images <= len(
        dataset
    ), "Too few images in the dataset."

    font = ImageFont.truetype(str(Path(__file__).parent.parent / "assets/Roboto-Black.ttf"), 40)

    images = []
    for _ in trange(FLAGS.num_images):
        idx = rng.choice(len(dataset))
        image, label = dataset[idx]
        if FLAGS.include_class_names:
            class_name = dataset.classes[label]
            image = ImageDraw.Draw(image)
            position = (25, 25)
            bbox = image.textbbox(position, class_name, font=font)
            image.rectangle(bbox, fill="white")
            image.text(position, class_name, font=font, fill="black")
            image = image._image
        images.append(T.ToTensor()(image))

    output_image = make_grid(images, nrow=int(np.sqrt(FLAGS.num_images)))
    output_image = T.ToPILImage()(output_image)
    output_image.save(FLAGS.output_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_path"])
    app.run(main)
