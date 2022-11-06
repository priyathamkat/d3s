import torch
import torchvision.transforms as T
import torch.nn as nn


class DINO(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main", self.arch, pretrained=True
        )
        linear_ckpt_urls = {
            "dino_vits16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth",
            "dino_vits8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth",
            "dino_vitb16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth",
            "dino_vitb8": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth",
            "dino_xcit_small_12_p16": "https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth",
            "dino_xcit_small_12_p8": "https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth",
            "dino_xcit_medium_24_p16": "https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth",
            "dino_xcit_medium_24_p8": "https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth",
            "dino_resnet50": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth",
        }
        linear_ckpt = torch.hub.load_state_dict_from_url(
            linear_ckpt_urls[self.arch], map_location="cpu"
        )["state_dict"]
        modified_ckpt = {}
        for k, v in linear_ckpt.items():
            modified_ckpt[k.split(".")[-1]] = v
        self.fc = nn.Linear(*list(reversed(modified_ckpt["weight"].shape)))
        self.fc.load_state_dict(modified_ckpt)
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.n = 4 if "vits" in self.arch else 1
        self.avgpool = "vitb" in self.arch

    def forward(self, x):
        if "vit" in self.arch:
            intermediate_output = self.backbone.get_intermediate_layers(x, self.n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = self.backbone(x)
            
        output = self.fc(output)
        return output
