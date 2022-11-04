import torch.nn as nn
import torch


class MINELoss(nn.Module):
    def __init__(self, feature_dim=1024, ma_rate=0.1):
        super().__init__()
        linear_layers = []
        dim = feature_dim + 1
        while dim != 1:
            linear_layers.extend([nn.Linear(dim, dim // 2), nn.ELU()])
            dim = dim // 2
        self.T = nn.Sequential(
            *linear_layers,
        )
        self.ma_rate = ma_rate
        self.indenpendent_t_ma = 1

    def forward(self, x, y, optimize_T=False):
        y = y.unsqueeze(1)
        joint_xy = torch.cat([x, y], dim=1)
        joint_t = self.T(joint_xy).mean(dim=0)
        shuffled_y = y[torch.randperm(y.shape[0])]
        independent_xy = torch.cat([x, shuffled_y], dim=1)
        if optimize_T:
            exp_independent_t = self.T(independent_xy).exp().mean(dim=0)
            self.indenpendent_t_ma = (
                1 - self.ma_rate
            ) * self.indenpendent_t_ma + self.ma_rate * exp_independent_t
            loss = -(
                joint_t - (1 / self.indenpendent_t_ma).detach() * exp_independent_t
            )
        else:
            log_exp_independent_t = torch.log(self.T(independent_xy).exp().mean(dim=0))
            loss = -(joint_t - log_exp_independent_t)
            self.indenpendent_t_ma = 1
        return loss
