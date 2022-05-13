import math
from typing import Optional

import torch


class WarmupCosineLambda:
    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(self.decay_scale, -epoch / self.warmup_steps)
            ratio = epoch / self.warmup_steps
        else:
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio


def topk_average_precision(output: torch.Tensor, y: torch.Tensor, k: int):
    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)
    topk = output.topk(k)[1]
    match_mat = topk == y[:, None].expand(topk.shape)
    return (match_mat * score_array).sum(dim=1)


def calc_map5(output: torch.Tensor, y: torch.Tensor, threshold: Optional[float]):
    if threshold is not None:
        output = torch.cat([output, torch.full((output.shape[0], 1), threshold, device=output.device)], dim=1)
    return topk_average_precision(output, y, 5).mean().detach()


def map_dict(output: torch.Tensor, y: torch.Tensor, prefix: str):
    d = {f"{prefix}/acc": topk_average_precision(output, y, 1).mean().detach()}
    for threshold in [None, 0.3, 0.4, 0.5, 0.6, 0.7]:
        d[f"{prefix}/map{threshold}"] = calc_map5(output, y, threshold)
    return d
