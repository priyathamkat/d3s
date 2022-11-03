import json
from pathlib import Path

import boto3
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "../assets/input/inputData.json", "Path to inputData.json")

BUCKET_NAME = "d3s-bucket"
FOLDER_NAME = "d3s_samples"

def main(argv):
    client = boto3.client("s3")

    with open(FLAGS.input) as f:
        inputData = json.load(f)
    for datum in tqdm(inputData):
        image_path = Path(datum["image"])
        key = f"{FOLDER_NAME}/{Path(*image_path.parts[-3:])}"
        client.upload_file(str(image_path), BUCKET_NAME, key)

if __name__ == "__main__":
    app.run(main)