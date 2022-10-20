import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from absl import app, flags
from d3s.datasets import CocoDetection, ImageNet, SalientImageNet
from d3s.diffusion_generator import DiffusionGenerator
from d3s.input_generator import InputGenerator
from torchvision import transforms as T
from tqdm import trange

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "generate_from_any_class", False, "Generate each sample from any class randomly"
)
flags.DEFINE_integer("num_classes", 10, "Number of classes to generate from")
flags.DEFINE_integer("num_samples_per_class", 10, "Number of samples to generate")
flags.DEFINE_enum(
    "dataset",
    "imagenet",
    ["imagenet", "salient-imagenet", "coco"],
    "Dataset to generate samples for",
)
flags.DEFINE_float("strength", 0.9, "Noise strength for diffusion model")
flags.DEFINE_string("output_folder", None, "Output folder for generated images")
flags.DEFINE_list("template_files", None, "List of prompt template files")
flags.DEFINE_bool(
    "use_foreground_image", True, "Use foreground image for initialization"
)
flags.DEFINE_bool(
    "use_background_image", True, "Use background image for initialization"
)
flags.DEFINE_bool("use_mask", False, "Use masked images for initialization")
flags.DEFINE_float("fg_scale", 0.4, "Scale of foreground image to background")
flags.DEFINE_bool("save_init", False, "Save init_image along with generated image")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs to use")
flags.DEFINE_integer("seed", 8432, "Base seed for diffusion model")

flags.register_validator(
    "use_mask",
    lambda value: FLAGS.use_foreground_image or not value,
    "Cannot use mask without foreground image",
)
flags.register_validator(
    "use_mask",
    lambda value: FLAGS.dataset != "imagenet" or not value,
    "Cannot use mask with ImageNet",
)
flags.register_validator(
    "save_init",
    lambda value: FLAGS.use_foreground_image or FLAGS.use_background_image or not value,
    "Cannot save init image without foreground image or background image",
)

rng = np.random.default_rng(7245)


def generate(rank, queue, lock):
    device = torch.device(f"cuda:{rank}")
    lock.acquire()
    try:
        diffusion = DiffusionGenerator(seed=rank + FLAGS.seed, device=device)
    finally:
        lock.release()

    while True:
        text_prompt, init_image, save_path = queue.get()

        if init_image is not None:
            init_image = init_image.to(device)
            if text_prompt is not None:
                generated_image = diffusion.conditional_generate(
                    text_prompt, init_image, FLAGS.strength, return_init=FLAGS.save_init
                )
            else:
                generated_image = diffusion.conditional_generate(
                    "", init_image, FLAGS.strength, return_init=FLAGS.save_init
                )
        else:
            if text_prompt is not None:
                generated_image = diffusion.unconditional_generate(text_prompt)
            else:
                break  # encountered sentinel values

        generated_image.save(save_path)
        del text_prompt, init_image, save_path


def main(argv):
    if FLAGS.dataset == "imagenet":
        dataset = ImageNet()
    elif FLAGS.dataset == "salient-imagenet":
        dataset = SalientImageNet()
    else:
        dataset = CocoDetection()

    input_generators = [
        InputGenerator(dataset, template_file) for template_file in FLAGS.template_files
    ]

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            T.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )

    output_folder = Path(FLAGS.output_folder)
    shift_folders = [
        output_folder / f"{input_generator.shift_name}"
        for input_generator in input_generators
    ]
    for shift_folder in shift_folders:
        shift_folder.mkdir(exist_ok=True, parents=True)
    FLAGS.append_flags_into_file(output_folder / "flags.txt")

    metadata = []

    num_samples = FLAGS.num_classes * FLAGS.num_samples_per_class
    classes_to_generate = rng.choice(
        len(dataset.classes), size=FLAGS.num_classes, replace=False
    )

    queue = mp.Queue(maxsize=3 * FLAGS.num_gpus)
    lock = mp.Lock()
    processes = []
    for rank in reversed(range(FLAGS.num_gpus)):
        p = mp.Process(target=generate, args=(rank, queue, lock))
        p.start()
        processes.append(p)

    with trange(num_samples) as pbar:
        for i in range(FLAGS.num_classes):
            for j in range(FLAGS.num_samples_per_class):
                if FLAGS.generate_from_any_class:
                    # a random class is chosen for each sample
                    class_idx = rng.choice(len(dataset.classes))
                else:
                    class_idx = classes_to_generate[i]

                image_input = None
                image_args = {}

                for input_generator, shift_folder in zip(
                    input_generators, shift_folders
                ):
                    if image_input is None and (
                        FLAGS.use_foreground_image or FLAGS.use_background_image
                    ):
                        image_input, image_args = input_generator.generate_image(
                            use_foreground_image=FLAGS.use_foreground_image,
                            use_background_image=FLAGS.use_background_image,
                            use_mask=FLAGS.use_mask,
                            fg_scale=FLAGS.fg_scale,
                            class_idx=class_idx,
                        )
                    if image_args:
                        text_prompt, prompt_args = input_generator.generate_prompt(
                            **image_args,
                        )
                    else:
                        text_prompt, prompt_args = input_generator.generate_prompt(
                            class_idx=class_idx
                        )

                    init_image = None
                    if image_input is not None:
                        if "background" in prompt_args:
                            init_image = image_transform(image_input.init_image)
                        else:
                            init_image = image_transform(
                                replace(image_input, bg_image=None).init_image
                            )

                    args = {**image_args, **prompt_args}
                    save_name = str(shift_folder / f"{i * FLAGS.num_classes + j}.png")
                    queue.put((text_prompt, init_image, save_name))

                    metadatum = {
                        "image": save_name,
                        "classIdx": class_idx,
                        "prompt": text_prompt,
                        "args": args,
                    }
                    metadatum["hasInit"] = FLAGS.save_init
                    try:
                        metadatum["background"] = args["background"].split(" ")[-1]
                    except KeyError:
                        pass
                    metadata.append(metadatum)

                pbar.update(1)

    for _ in range(FLAGS.num_gpus):
        queue.put((None, None, None))  # sentinel values to signal subprocesses to exit

    for p in processes:
        p.join()  # wait for all subprocesses to finish

    with open(output_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, default=str)


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_folder", "template_files"])
    app.run(main)
