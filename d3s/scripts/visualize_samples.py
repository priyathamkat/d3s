from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.font_manager import FontProperties, fontManager
from tqdm import tqdm

from d3s.datasets import D3S

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "d3s_root",
    "/cmlscratch/pkattaki/datasets/d3s",
    "Path to D3S root directory. This should contain metadata.json",
)
flags.DEFINE_enum(
    "shift",
    "background-shift",
    ["all", "background-shift", "geography-shift", "time-shift"],
    "Shift to use for evaluation",
)
flags.DEFINE_string("output_folder", None, "Path of the output image.")
flags.DEFINE_bool(
    "include_class_names",
    True,
    "Whether to include the class names in the output image.",
)
flags.DEFINE_bool(
    "return_init",
    True,
    "Whether to return the initial image along with the generated image.",
)


def main(argv):
    rng = np.random.default_rng(0)
    dataset = D3S(Path(FLAGS.d3s_root), return_init=FLAGS.return_init, shift=FLAGS.shift)

    font_path = "/cmlscratch/pkattaki/void/d3s/d3s/assets/Roboto-Regular.ttf"
    fontManager.addfont(font_path)
    prop = FontProperties(fname=font_path, weight="regular")

    print(prop)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()

    class_idxs = [360, 853]

    output_folder = Path(FLAGS.output_folder)
    for i, class_idx in enumerate(tqdm(class_idxs)):
        idx = rng.choice(dataset.class_to_indices[class_idx])
        image, _, _ = dataset[idx]

        _ = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_folder / f"io-{i}.png")
        print(dataset.metadata[idx]["prompt"])


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_folder"])
    app.run(main)
