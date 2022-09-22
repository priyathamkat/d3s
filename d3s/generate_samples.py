import logging
from pathlib import Path
from typing import List

import numpy as np
from absl import app, flags
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as T
from tqdm import trange

from diffusion_generator import DiffusionGenerator
from salient_imagenet import MTurkResults, SalientImageNet
from utils import paste_on_bg

logging.getLogger("seed").setLevel(logging.CRITICAL)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_samples", 10, "Number of samples to generate")
flags.DEFINE_enum(
    "dataset",
    "imagenet",
    ["imagenet", "salient-imagenet", "coco"],
    "Dataset to generate samples for",
)
flags.DEFINE_bool("use_mask", False, "Use masked images for initialization")
flags.DEFINE_float("fg_scale", 0.4, "Scale of foreground image to background")
flags.DEFINE_float("strength", 0.9, "Noise strength for diffusion model")

rng = np.random.default_rng()


class FGImages:
    def __init__(self) -> None:
        mturk_results = MTurkResults()
        self.core_features_dict = mturk_results.core_features_dict

    def get_images(self, dataset, class_idx, use_mask=False):
        if dataset == "salient-imagenet":
            dataset = SalientImageNet(class_idx, self.core_features_dict[class_idx])
            for image, mask in dataset:
                if use_mask:
                    image = image * mask
                image = T.ToPILImage()(image)
                yield image, mask
        else:
            raise NotImplementedError


def get_class_list(dataset):
    if dataset == "salient-imagenet" or dataset == "imagenet":
        with open("./imagenet_classes.txt", "r") as f:
            class_list = eval(f.read())
    else:
        raise NotImplementedError

    return class_list


def get_background(bg_prompt, images_folder):
    for subfolder in images_folder.iterdir():
        if subfolder.name in bg_prompt:
            return rng.choice([p for p in subfolder.iterdir() if p.suffix == ".jpeg"])


def get_random_backgrounds(num_samples, bg_prompt_templates, images_folder):
    bgs = []
    bg_prompts = []
    for _ in range(num_samples):
        bg_prompt = rng.choice(bg_prompt_templates)
        bg_prompts.append(bg_prompt)
        bgs.append(get_background(bg_prompt, images_folder))
    return bg_prompts, bgs


def main(argv):

    diffusion = DiffusionGenerator()
    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    classes = get_class_list(FLAGS.dataset)
    fg_images = FGImages()

    images_folder = Path("./images/backgrounds/")
    outputs_folder = Path("./images/outputs")

    with open("./bg_prompt_templates.txt", "r") as f:
        bg_prompt_templates = f.readlines()

    num_classes = FLAGS.num_samples // len(bg_prompt_templates)
    px = 1 / plt.rcParams["figure.dpi"]
    nrows = num_classes
    ncols = len(bg_prompt_templates)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=(400 * (ncols + 1) * px, 200 * (nrows + 1) * px),
    )

    for ax, bg_prompt in zip(axes[0], bg_prompt_templates):
        ax.set_title(bg_prompt)

    with trange(FLAGS.num_samples) as pbar:
        for i in range(num_classes):
            class_idx = rng.choice(len(classes))
            class_name = classes[class_idx]
            class_name = class_name.split(",")[0]
            axes[i, 0].set_ylabel(class_name, rotation=90, size="large")
            fgs = iter(fg_images.get_images(FLAGS.dataset, class_idx, FLAGS.use_mask))
            for j, ((fg, mask), bg_prompt) in enumerate(zip(fgs, bg_prompt_templates)):
                bg = get_background(bg_prompt, images_folder)
                bg = Image.open(bg)
                if FLAGS.dataset == "salient-imagenet":
                    init_image = paste_on_bg(fg, bg, size=512, fg_scale=FLAGS.fg_scale)
                    init_image = image_transform(init_image).unsqueeze(0)
                    prompt = f"a photo of a {class_name} {bg_prompt}"
                    output = diffusion.conditional_generate(
                        prompt, init_image, FLAGS.strength
                    )
                    output.save(outputs_folder / f"{prompt}.png")
                    axes[i, j].imshow(output.resize((400, 200), Image.Resampling.BICUBIC))
                else:
                    raise NotImplementedError
                pbar.update(1)
    fig.tight_layout()
    plt.savefig(outputs_folder / "outputs.png")


if __name__ == "__main__":
    app.run(main)
