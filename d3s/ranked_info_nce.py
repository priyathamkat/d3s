import torch
import torch.nn as nn


class RankedInfoNCE(nn.Module):
    def __init__(self, t: float = 1.0) -> None:
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=2)
        self.t = t

    def forward(self, query, results, alphas):
        query.unsqueeze_(1)
        exp_sims = torch.exp(self.cos(query, results) / self.t).sum(dim=0)
        cum_exp_sims = torch.cumsum(exp_sims, dim=0)
        loss = -torch.dot(torch.log(exp_sims[1:] / cum_exp_sims[1:]), alphas)
        return loss
