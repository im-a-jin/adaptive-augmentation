import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class MultivariateNormalNoise:
    def __init__(self, loc, covariance_matrix):
        self.m = MultivariateNormal(loc, covariance_matrix)

    def __call__(self, x):
        # https://bochang.me/blog/posts/pytorch-distributions/
        return x + self.m.sample(torch.Size([len(x)]))


class Clamp:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        return torch.clamp(x, min=self.lower, max=self.upper)
