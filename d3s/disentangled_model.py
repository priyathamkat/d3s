import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledModel(nn.Module):
    def __init__(
        self,
        model,
        num_fg_features=None,
        num_bg_features=None,
        num_fg_classes=1000,
        num_bg_classes=10,
    ):
        super().__init__()

        self.model = model
        self.num_features = model.fc.in_features

        assert (num_fg_features is not None) ^ (
            num_bg_features is not None
        ), "Must specify exactly one of num_bg_features or num_fg_features"
        if num_bg_features is None:
            num_bg_features = self.num_features - num_fg_features
        if num_fg_features is None:
            num_fg_features = self.num_features - num_bg_features

        self.num_fg_features = num_fg_features
        self.num_bg_features = num_bg_features

        self.fg_fc = nn.Linear(self.num_fg_features, num_fg_classes)
        with torch.no_grad():
            self.fg_fc.weight.copy_(model.fc.weight[:, :num_fg_features])
            self.fg_fc.bias.copy_(model.fc.bias)
        self.bg_fc = nn.Linear(self.num_bg_features, num_bg_classes)
        self.model.fc = nn.Identity()

    def forward(self, x):
        features = self.model(x)
        fg_features = features[:, : self.num_fg_features]
        bg_features = features[:, -self.num_bg_features :]

        fg_outputs = self.fg_fc(fg_features)
        bg_outputs = self.bg_fc(bg_features)

        return fg_outputs, bg_outputs, fg_features, bg_features
