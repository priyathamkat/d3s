import torch
import numpy as np
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "random seed")

def main(argv):
	np.random.seed(FLAGS.seed)

	perm = np.random.choice(10, size = 9, replace = False)

	torch.save(perm, "label_permutation.pth")

if __name__ == "__main__":
    app.run(main)