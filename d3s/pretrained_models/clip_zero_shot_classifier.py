import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torchvision.transforms as T


class CLIPZeroShotClassifier(nn.Module):
    def __init__(self, arch, device):
        super().__init__()
        arch = arch.replace("clip-", "")
        self.model, preprocess = clip.load(arch, device=device)
        self.transform = T.Compose(
            [
                T.Lambda(lambda x: preprocess(x)),
                T.Lambda(lambda x: x.unsqueeze(0)),
            ]
        )
        with open(
            Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r"
        ) as f:
            text = [v.split(",")[0] for v in json.load(f).values()]
        tokens = torch.cat([clip.tokenize(f"a photo of a {t}") for t in text]).to(
            device
        )
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return similarity
