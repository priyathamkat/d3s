import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import FlavaModel, AutoTokenizer, FlavaFeatureExtractor


class FlavaZeroShotClassifier(nn.Module):
    def __init__(self, arch, device):
        super().__init__()
        self.flava_model = FlavaModel.from_pretrained(arch)
        self.flava_feature_extractor = FlavaFeatureExtractor.from_pretrained(arch)
        self.transform = None
        self.device = device

        self.flava_model.to(self.device)

        with open(
            Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r"
        ) as f:
            text = [v.split(",")[0] for v in json.load(f).values()]

            text_prompts = [f"a photo of a {t}" for t in text]
        tokenizer = AutoTokenizer.from_pretrained(arch)

        with torch.no_grad():
            text_inputs = tokenizer(
                text_prompts, max_length=77, padding=True, return_tensors="pt"
            ).to(self.device)
            self.text_embeddings = self.flava_model.get_text_features(**text_inputs)[
                :, 0, :
            ]
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)

    def forward(self, x):
        image_features = self.flava_feature_extractor(x, return_tensors="pt").to(
            self.device
        )
        image_embeddings = self.flava_model.get_image_features(**image_features)[
            :, 0, :
        ]
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_embeddings @ self.text_embeddings.t()

        return logits
