import json
import logging
from pathlib import Path

import numpy as np
from absl import app, flags
from torchvision import transforms as T
from tqdm import trange

from d3s.datasets import CocoDetection, ImageNet, SalientImageNet
from d3s.diffusion_generator import DiffusionGenerator
from d3s.input_generator import InputGenerator

logging.getLogger("seed").setLevel(logging.CRITICAL)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_samples", 10, "Number of samples to generate")
flags.DEFINE_enum(
    "dataset",
    "imagenet",
    ["imagenet", "salient-imagenet", "coco"],
    "Dataset to generate samples for",
)
flags.DEFINE_float("strength", 0.9, "Noise strength for diffusion model")
flags.DEFINE_string("output_folder", None, "Output folder for generated images")
flags.DEFINE_string("template_file", None, "File containing prompt template")
flags.DEFINE_bool("use_init_image", True, "Use init image for initialization")
flags.DEFINE_bool(
    "use_background_image", True, "Use background image for initialization"
)
flags.DEFINE_bool("use_mask", False, "Use masked images for initialization")
flags.DEFINE_float("fg_scale", 0.4, "Scale of foreground image to background")
flags.DEFINE_bool("save_init", False, "Save init_image along with generated image")

flags.register_validator(
    "use_background_image",
    lambda value: FLAGS.use_init_image or not value,
    "Cannot use background image without init image",
)
flags.register_validator(
    "use_mask",
    lambda value: FLAGS.use_init_image or not value,
    "Cannot use mask without init image",
)
flags.register_validator(
    "use_mask",
    lambda value: FLAGS.dataset != "imagenet" or not value,
    "Cannot use mask with ImageNet",
)

rng = np.random.default_rng()


def main(argv):
    if FLAGS.dataset == "imagenet":
        dataset = ImageNet()
    elif FLAGS.dataset == "salient-imagenet":
        dataset = SalientImageNet()
    else:
        dataset = CocoDetection()

    input_generator = InputGenerator(dataset, FLAGS.template_file)

    diffusion = DiffusionGenerator()

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            T.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )

    outputs_folder = Path(FLAGS.output_folder)
    outputs_folder.mkdir(exist_ok=True)
    FLAGS.append_flags_into_file(outputs_folder / "flags.txt")

    metadata = {}

    for i in trange(FLAGS.num_samples):
        if FLAGS.use_init_image:
            prompt, init_image, args = input_generator.generate_input(
                use_background_image=FLAGS.use_background_image,
                use_mask=FLAGS.use_mask,
                fg_scale=FLAGS.fg_scale,
            )
            init_image = image_transform(init_image)
            generated_image = diffusion.conditional_generate(
                prompt, init_image, FLAGS.strength, return_init=FLAGS.save_init
            )
        else:
            prompt, args = input_generator.generate_prompt()
            generated_image = diffusion.unconditional_generate(prompt, FLAGS.strength)
        save_name = f"{i}.png"
        generated_image.save(outputs_folder / save_name)
        metadata[save_name] = {"prompt": prompt, "args": args}

    with open(outputs_folder / "metadata.txt", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    app.run(main)