import json
from pathlib import Path

import requests
from tqdm import tqdm

with open(Path(__file__).parents[1] / "metadata/imagenet_label_to_wordnet_synset.txt", "r") as f:
    imagenet_label_to_wordnet_synset = eval(f.read())

dictionary = {}
for class_idx in tqdm(imagenet_label_to_wordnet_synset):
    class_data = imagenet_label_to_wordnet_synset[class_idx]
    uri = class_data["uri"]
    uri = uri.replace("wn30", "json/pwn30")
    r = requests.get(uri)
    data = json.loads(r.text)
    dictionary[str(class_idx)] = data[0]["definition"]

with open(Path(__file__).parents[1] / "metadata/imagenet_dictionary.json", "w") as f:
    json.dump(dictionary, f)
