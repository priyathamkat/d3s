import sys
from itertools import product
from pathlib import Path

import lpips
import numpy as np
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "images_dir", None, "Directory containing images to compute diversity for"
)
flags.DEFINE_integer("num_samples", 10, "Number of samples to compute diversity for")


def main(argv):
    rng = np.random.default_rng()
    images_dir = Path(FLAGS.images_dir)
    images = list(images_dir.glob("*.png"))
    pairs = list(product(images, repeat=2))
    samples = rng.choice(pairs, size=FLAGS.num_samples, replace=False)
    distances = np.zeros(FLAGS.num_samples)
    lpips_distance = lpips.LPIPS(net="alex").cuda()
    for i, sample in enumerate(tqdm(samples, file=sys.stdout)):
        img1, img2 = sample
        img1 = lpips.im2tensor(lpips.load_image(str(img1))).cuda()
        img2 = lpips.im2tensor(lpips.load_image(str(img2))).cuda()

        distances[i] = lpips_distance.forward(img1, img2).item()

    print(
        f"Diversity (LPIPS distance): {distances.mean():.4f} \u00B1 {distances.std():.4f}"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["images_dir"])
    app.run(main)
