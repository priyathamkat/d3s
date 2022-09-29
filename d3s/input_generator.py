from pathlib import Path
from string import Formatter, Template
from typing import Any, Tuple, Union

import numpy as np
import yaml
from PIL import Image

from d3s.datasets.coco import CocoDetection
from d3s.datasets.salient_imagenet import SalientImageNet
from d3s.utils import crop_to_squarize, paste_on_bg, resize

rng = np.random.default_rng()


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
            self.prompt_template = Template(
                yaml.load(f, Loader=yaml.CLoader)["template"]
            )
            # parse all identifiers in the template
            self.prompt_identifiers = {
                e[1]
                for e in Formatter().parse(self.prompt_template.template)
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
            if "definition" in self.prompt_identifiers:
                substitutes["definition"] = self.dataset.dictionary[class_idx]

        for identifier in self.prompt_identifiers - {"class_name", "definition"}:
            try:
                substitutes[identifier] = kwargs[identifier]
            except KeyError:
                substitutes[identifier] = rng.choice(self.prompt_options[identifier])
                kwargs[identifier] = substitutes[identifier]
        return self.prompt_template.substitute(substitutes), kwargs

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

        fg = None
        mask = None
        if use_foreground_image:
            try:
                class_idx = kwargs["class_idx"]
            except KeyError:
                class_idx = rng.choice(len(self.dataset.classes))
                kwargs["class_idx"] = class_idx
            finally:
                dataset_item = self.dataset.get_random(class_idx)

            fg = dataset_item[0]
            if use_mask:
                if isinstance(self.dataset, SalientImageNet):
                    fg, mask = dataset_item
                elif (
                    isinstance(self.dataset, CocoDetection)
                    and self.dataset.catId is not None
                ):
                    fg, _, mask = dataset_item
                else:
                    raise ValueError(
                        f"{type(self.dataset).__name__} does not have masks"
                    )

        if use_background_image:
            try:
                bg = kwargs["background"]
            except KeyError:
                bg = rng.choice(self.prompt_options["background"])
                kwargs["background"] = bg
            finally:
                for background_folder in self.background_folders:
                    if background_folder.name in bg:
                        bg = rng.choice(list(background_folder.iterdir()))
                        break
            bg = Image.open(bg)

        if fg is None:
            # Resize background image to 512x512
            bg = crop_to_squarize(bg)
            bg = resize(bg, 512)
            return bg
        elif bg is None:
            # Resize foreground image to 512x512
            fg = crop_to_squarize(fg)
            fg = resize(fg, 512)
            return fg
        else:
            if "fg_scale" not in kwargs:
                kwargs["fg_scale"] = 0.4  # default `fg_scale`
            return paste_on_bg(fg, bg, mask=mask, fg_scale=kwargs["fg_scale"])

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

        prompt, args = self.generate_prompt(**kwargs)
        init_image = self.generate_image(
            use_foreground_image=use_foreground_image,
            use_background_image=use_background_image,
            use_mask=use_mask,
            **kwargs,
        )

        return prompt, init_image, args


if __name__ == "__main__":
    from datasets.imagenet import ImageNet

    input_generator = InputGenerator(
        ImageNet(),
        "/cmlscratch/pkattaki/void/d3s/d3s/recipes/prompts/photo-definition-background.yaml",
    )
    prompt, image, _ = input_generator.generate_input(
        class_idx=100,
        use_background_image=True,
        use_mask=False,
    )
    print(prompt)
    image.save("test.png")
