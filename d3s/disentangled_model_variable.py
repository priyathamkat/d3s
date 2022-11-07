import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SeparableLinear(nn.Module):
    def __init__(self, num_features, weights_a, weights_b):
        super().__init__()
        self.num_features = num_features
        self.split = weights_a.shape[1]
        combined_matrix = torch.cat([weights_a,weights_b],dim=1)
        # Confirm invertability; will except otherwise
        np.linalg.inv(combined_matrix.cpu().numpy().T)
        self.weight = nn.Parameter(combined_matrix)


    def forward(self, x):

        features = F.linear(x, self.weight, None)
        return (
            features[:, : self.split],
            features[:, self.split :],
        )


class DisentangledModelVariable(nn.Module):
    def __init__(self, model, weights_a, weights_b):
        super().__init__()
        self.model = model
        self.fc = self.model.fc
        self.disentangle = SeparableLinear(
            self.fc.in_features, weights_a, weights_b
        )
        self.model.fc = nn.Identity()
    def forward(self, x):
        features = self.model(x)
        fg_features, bg_features = self.disentangle(features)
        outputs = self.fc(features)
        return outputs, fg_features, bg_features