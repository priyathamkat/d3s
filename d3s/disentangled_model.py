import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleLinear(nn.Module):
    def __init__(self, num_features, lu_decompose=True, bias=False):
        super().__init__()
        self.num_features = num_features
        self.lu_decompose = lu_decompose

        if self.lu_decompose:
            w_init, _ = torch.linalg.qr(
                torch.randn(self.num_features, self.num_features)
            )
            p, lower, upper = torch.linalg.lu(w_init)
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, diagonal=1)
            l_mask = torch.tril(torch.ones_like(w_init), diagonal=-1)
            eye = torch.eye(self.num_features)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("eye", eye)
        else:
            weight = torch.empty(self.num_features, self.num_features)
            nn.init.xavier_normal_(weight)
            self.weight = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.zeros(self.num_features)) if bias else None

    def forward(self, x):
        if self.lu_decompose:
            lower = self.lower * self.l_mask + self.eye
            upper = self.upper * self.l_mask.t().contiguous()
            upper = upper + torch.diag(self.sign_s * torch.exp(self.log_s))
            weight = self.p @ lower @ upper
            features = F.linear(x, weight, self.bias)
        else:
            features = F.linear(x, self.weight, self.bias)
        return (
            features[:, : self.num_features // 2],
            features[:, self.num_features // 2 :],
        )


class DisentangledModel(nn.Module):
    def __init__(self, model, lu_decompose=True):
        super().__init__()
        self.model = model
        self.fc = self.model.fc
        self.disentangle = InvertibleLinear(
            self.fc.in_features, lu_decompose=lu_decompose
        )
        self.model.fc = nn.Identity()

    def forward(self, x):
        features = self.model(x)
        fg_features, bg_features = self.disentangle(features)
        outputs = self.fc(features)
        return outputs, fg_features, bg_features