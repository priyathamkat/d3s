import sys
from itertools import product
from pathlib import Path

import lpips
import numpy as np
from absl import app, flags
from d3s.datasets import D3S
from tqdm import trange

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "images_dir", None, "Directory containing images to compute diversity for"
)
flags.DEFINE_integer(
    "num_samples_per_class", 10, "Number of samples per class to compute diversity for"
)


def main(argv):
    rng = np.random.default_rng()
    dataset = D3S(Path(FLAGS.images_dir))
    lpips_distance = lpips.LPIPS(net="alex").cuda()
    num_samples = FLAGS.num_samples_per_class * len(dataset.class_to_indices)
    distances = np.zeros(num_samples)
    i = 0
    with trange(num_samples, file=sys.stdout) as pbar:
        for _, indices in dataset.class_to_indices.items():
            image_paths = [dataset.images[idx]["image_path"] for idx in indices]
            pairs = list(product(image_paths, repeat=2))
            samples = rng.choice(pairs, size=FLAGS.num_samples_per_class)
            for sample in samples:
                img1, img2 = sample
                img1 = lpips.im2tensor(lpips.load_image(str(img1))).cuda()
                img2 = lpips.im2tensor(lpips.load_image(str(img2))).cuda()

                distances[i] = lpips_distance.forward(img1, img2).item()

                i += 1
                pbar.update(1)

    print(
        f"Diversity (LPIPS distance) of {FLAGS.images_dir}: {distances.mean():.4f} \u00B1 {distances.std():.4f}"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["images_dir"])
    app.run(main)
