from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter, Template
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image

from d3s.datasets.coco import CocoDetection
from d3s.datasets.salient_imagenet import SalientImageNet
from d3s.utils import crop_to_squarize, pad_to_squarize, paste_on_bg, resize

rng = np.random.default_rng()


@dataclass(repr=False, eq=False, frozen=True)
class ImageInput:
    fg_image: Optional[Image.Image] = None
    bg_image: Optional[Image.Image] = None
    mask: Optional[torch.Tensor] = None
    fg_scale: float = 0.4
    init_image: Image.Image = field(init=False)

    def __post_init__(self):
        mask = self.mask.numpy() if self.mask is not None else None
        if self.fg_image is not None:
            if self.bg_image is not None:
                init_image = paste_on_bg(
                    self.fg_image,
                    self.bg_image,
                    mask=mask,
                    fg_scale=self.fg_scale,
                )
            else:
                if self.mask is not None:
                    init_image = pad_to_squarize(np.array(self.fg_image) * mask)
                    init_image = resize(Image.fromarray(self.init_image), 512)
                else:
                    init_image = resize(crop_to_squarize(self.fg_image), 512)
        else:
            if self.bg_image is not None:
                init_image = resize(crop_to_squarize(self.bg_image), 512)
            else:
                raise ValueError(
                    "At least one of `fg_image` and `bg_image` must not be None."
                )
        # use __setattr__ to circumvent FrozenInstanceError
        object.__setattr__(self, "init_image", init_image)


class InputGenerator:
    """Generates a prompt and an image from a dataset to feed into a diffusion model.

    Args:
        dataset (Any): dataset to generate input from
        template_file (Union[str, Path]): template file to generate prompt from
    """

    def __init__(self, dataset: Any, template_file: Union[str, Path]) -> None:
        self.dataset = dataset

        with open(Path(template_file).resolve(), "r") as f:
            # load template file
            template_file = yaml.load(f, Loader=yaml.CLoader)

        self.prompt_templates = [Template(t) for t in template_file["template"]]
        self.shift_name = template_file["shift_name"]
        # parse all identifiers in the template
        self.prompt_identifiers = {}
        for prompt_template in self.prompt_templates:
            self.prompt_identifiers[prompt_template] = {
                e[1]
                for e in Formatter().parse(prompt_template.template)
                if e[1] is not None
            }

        # load all options for each identifier
        with open(Path(__file__).parent / "recipes/prompts/options.yaml", "r") as f:
            self.prompt_options = yaml.load(f, Loader=yaml.CLoader)[0]

        # load all background folders
        self.background_folders = list(
            (Path(__file__).parent / "images/backgrounds").iterdir()
        )

    def generate_prompt(self, **kwargs) -> Tuple[str, dict]:
        """
        Generate an prompt from the dataset.

        Args:
            class_idx (int, optional): class index of the image to be generated
            background (str, optional): background to be used

        Returns:
            prompt (str): prompt input
            args (dict): arguments used to generate the prompt
        """
        prompt_template = rng.choice(self.prompt_templates)
        prompt_identifiers = self.prompt_identifiers[prompt_template]
        substitutes = {}
        # pick a random `class_idx` if not provided
        try:
            class_idx = kwargs["class_idx"]
        except KeyError:
            class_idx = rng.choice(len(self.dataset.classes))
            kwargs["class_idx"] = class_idx
        finally:
            class_name = self.dataset.classes[class_idx]
            substitutes["class_name"] = class_name
            if "definition" in prompt_identifiers:
                substitutes["definition"] = self.dataset.dictionary[class_idx]

        for identifier in prompt_identifiers - {"class_name", "definition"}:
            try:
                substitutes[identifier] = kwargs[identifier]
            except KeyError:
                substitutes[identifier] = rng.choice(self.prompt_options[identifier])
                kwargs[identifier] = substitutes[identifier]
        return prompt_template.substitute(substitutes), kwargs

    def generate_image(
        self,
        use_foreground_image=False,
        use_background_image=False,
        use_mask=False,
        **kwargs,
    ) -> Any:
        """
        Generate an `init_image` from the dataset.

        Args:
            use_background_image (bool): whether to paste the image on a background image
            use_mask (bool): whether to use a mask with foreground image
            class_idx (int, optional): class index of the image to be generated
            background (str, optional): background to be used
        """
        # pick a random `class_idx` if not provided
        assert (
            use_foreground_image or use_background_image
        ), "At least one of foreground or background image should be used"
        assert (
            not use_mask or use_foreground_image
        ), "Mask can only be used with foreground image"

        fg_image = None
        mask = None
        if use_foreground_image:
            try:
                class_idx = kwargs["class_idx"]
            except KeyError:
                class_idx = rng.choice(len(self.dataset.classes))
                kwargs["class_idx"] = class_idx
            finally:
                dataset_item = self.dataset.get_random(class_idx)

            fg_image = dataset_item[0]
            if use_mask:
                if isinstance(self.dataset, SalientImageNet):
                    fg_image, mask = dataset_item
                elif (
                    isinstance(self.dataset, CocoDetection)
                    and self.dataset.catId is not None
                ):
                    fg_image, _, mask = dataset_item
                else:
                    raise ValueError(
                        f"{type(self.dataset).__name__} does not have masks"
                    )

        bg_image = None
        if use_background_image:
            try:
                background = kwargs["background"]
            except KeyError:
                background = rng.choice(self.prompt_options["background"])
                kwargs["background"] = background
            finally:
                for background_folder in self.background_folders:
                    if background_folder.name in background:
                        background_path = rng.choice(list(background_folder.iterdir()))
                        break
            bg_image = Image.open(background_path)

        if "fg_scale" not in kwargs:
            kwargs["fg_scale"] = 0.4  # default `fg_scale`

        return (
            ImageInput(
                fg_image=fg_image,
                bg_image=bg_image,
                mask=mask,
                fg_scale=kwargs["fg_scale"],
            ),
            kwargs,
        )

    def generate_input(
        self,
        use_foreground_image=False,
        use_background_image=False,
        use_mask=False,
        **kwargs,
    ) -> Any:
        """
        Generate a `prompt` and an `init_image` from the dataset.

        Args:
            use_background_image (bool): whether to paste the image on a background image
            use_mask (bool): whether to use a mask with foreground image
            class_idx (int, optional): class index of the image to be generated
            background (str, optional): background to be used

        Returns:
            prompt (str): prompt input
            init_image (PIL.Image): init image input
            args (dict): arguments used to generate the prompt and the image
        """
        # pick a random `class_idx` if not provided
        if "class_idx" not in kwargs:
            kwargs["class_idx"] = rng.choice(len(self.dataset.classes))

        # pick a random `background` if not provided
        if use_background_image and "background" not in kwargs:
            kwargs["background"] = rng.choice(self.prompt_options["background"])

        text_prompt, prompt_args = self.generate_prompt(**kwargs)
        image_input, image_args = self.generate_image(
            use_foreground_image=use_foreground_image,
            use_background_image=use_background_image,
            use_mask=use_mask,
            **kwargs,
        )
        args = {**prompt_args, **image_args}
        return text_prompt, image_input, args


if __name__ == "__main__":
    from datasets.imagenet import ImageNet

    input_generator = InputGenerator(
        ImageNet(),
        Path(__file__).parent / "recipes/prompts/background-shift.yaml",
    )
    text_prompt, image_input, _ = input_generator.generate_input(
        class_idx=100,
        use_foreground_image=True,
        use_background_image=True,
        use_mask=False,
    )
    print(text_prompt)
    image_input.init_image.save("test.png")
