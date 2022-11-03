import csv
import json
from pathlib import Path

import numpy as np
from absl import app, flags
from tqdm import tqdm, trange

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "../assets/input/inputData.json", "Path to inputData.json")
flags.DEFINE_integer("batch_size", 25, "Non ground truth batch size for each AMT task")
flags.DEFINE_string("output", "../assets/amt-batches", "Path to output directory")

def extract_s3_path(path):
    return Path(*Path(path).parts[-3:])

def main(argv):
    rng = np.random.default_rng()
    FLAGS.output = Path(FLAGS.output)
    with open(FLAGS.input) as f:
        inputData = json.load(f)
    rng.shuffle(inputData)
    with open(FLAGS.output / "input_amt_batches.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["inputData"])
        writer.writeheader()
        
        for idx in trange(0, len(inputData), FLAGS.batch_size, desc="Writing AMT batches"):
            row = ""
            for datum in inputData[idx:idx + FLAGS.batch_size]:    
                image_path = str(extract_s3_path(datum["image"]))
                classIdx = datum["classIdx"]
                row += f"{image_path}, {classIdx}, "
            row = row[:-2]
            writer.writerow({
                "inputData": row,
            })
     

if __name__ == "__main__":
    app.run(main)