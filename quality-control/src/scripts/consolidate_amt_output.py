import csv
import json
from collections import defaultdict
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("amt_results", None, "Path to AMT results file")
flags.DEFINE_string("val_metadata", None, "Path to D3S val split metadata file")

ANNOTATIONS = "annotations"
ANSWERS = "Answer.taskAnswers"
FOREGROUND = "foreground"
NSFW = "nsfw"
YES = "Yes"

def extract_rel_path(path):
    return str(Path(*Path(path).parts[-2:]))


def main(argv):
    annotations = defaultdict(lambda: {FOREGROUND: 0, NSFW: 0})
    with open(Path(FLAGS.amt_results), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_annotations = json.loads(json.loads(row[ANSWERS])[0][ANNOTATIONS])
            for image, image_annotations in row_annotations.items():
                annotations[image][FOREGROUND] += image_annotations[FOREGROUND] == YES
                annotations[image][NSFW] += image_annotations[NSFW] == YES

    num_foreground = 0
    num_nsfw = 0
    filtered_images = []
    for image, attributes in annotations.items():
        if attributes[NSFW] > 0:
            num_nsfw += 1
        if attributes[FOREGROUND] == 5:
            num_foreground += 1
        if not (attributes[FOREGROUND] == 5 and attributes[NSFW] == 0):
            filtered_images.append(extract_rel_path(image))
    num_accepted = len(annotations) - len(filtered_images)

    print(
        f"Number of images with correct foreground: {num_foreground} ({100 * num_foreground / len(annotations):.2f})"
    )
    print(
        f"Number of NSFW images: {num_nsfw} ({100 * num_nsfw / len(annotations):.2f})"
    )
    print(
        f"Number of accepted images: {num_accepted} ({100 * num_accepted / len(annotations):.2f})"
    )

    FLAGS.val_metadata = Path(FLAGS.val_metadata)
    with open(FLAGS.val_metadata, "r") as f:
        metadata = json.load(f)
    filtered_metadata = list(filter(lambda metadatum: extract_rel_path(metadatum["image"]) not in filtered_images, metadata))

    with open(FLAGS.val_metadata.parent / "filtered_metadata.json", "w") as f:
        json.dump(filtered_metadata, f)


if __name__ == "__main__":
    app.run(main)
