import json

import requests
from tqdm import tqdm

with open("../txt_data/imagenet_label_to_wordnet_synset.txt", "r") as f:
    imagenet_label_to_wordnet_synset = eval(f.read())

dictionary = {}
for class_idx in tqdm(imagenet_label_to_wordnet_synset):
    class_data = imagenet_label_to_wordnet_synset[class_idx]
    uri = class_data["uri"]
    uri = uri.replace("wn30", "json/pwn30")
    r = requests.get(uri)
    data = json.loads(r.text)
    dictionary[class_idx] = data[0]["definition"]

with open("../txt_data/imagenet_dictionary.txt", "w") as f:
    f.write("{\n")
    for class_idx in dictionary:
        f.write(f"    {class_idx}: {dictionary[class_idx]},\n")
    f.write("}")