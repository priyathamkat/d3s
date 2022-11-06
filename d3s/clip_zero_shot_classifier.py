import json
from pathlib import Path

import clip
import torch.nn as nn


class CLIPZeroShotClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32")
        with open(
            Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r"
        ) as f:
            text = [v.split(",")[0] for v in json.load(f).values()]
        tokens = clip.tokenize(text)
        self.register_buffer("text_features", self.model.encode_text(tokens))

    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_logits, _ = self.model(image_features, self.text_features)
        return image_logits