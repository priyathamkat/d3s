from pathlib import Path

import torch
import torch.nn as nn
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

from d3s.constants import IMAGENET_PATH


class RobustResNet50(nn.Module):
    def __init__(self, arch):
        super().__init__()
        imagenet = ImageNet(str(IMAGENET_PATH))
        if arch == "robust_resnet50_l2":
            ckpt_name = "robust_resnet_l2_e3.0.pt"
        elif arch == "robust_resnet50_linf":
            ckpt_name = "robust_resnet_linf_e4_255.pt"
        else:
            raise ValueError(f"Unknown arch: {arch}")
        ckpt_path = str(Path(torch.hub.get_dir()) / "robustness" / ckpt_name)
        self.model, _ = make_and_restore_model(arch="resnet50", dataset=imagenet, resume_path=ckpt_path)
        
    def forward(self, x):
        x = self.model(x, with_image=False)
        return x