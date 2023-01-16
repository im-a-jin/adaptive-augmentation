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
    def __init__(self, loc, covariance_matrix, f=lambda x: x):
        self.f = f
        self.m = MultivariateNormal(loc, covariance_matrix)

    def __call__(self, x):
        # https://bochang.me/blog/posts/pytorch-distributions/#row-4
        noise = self.f(self.m.sample(torch.Size([])))
        return x + noise


class Mask:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        mask = torch.rand(x.shape) < (1 - self.p)
        return x * mask


class SparseNoise:
    def __init__(self, p, m=1):
        self.p = p
        self.m = m
        
    def __call__(self, x):
        mask = torch.rand(x.shape) < self.p
        return x + self.m * mask


class SaltAndPepper:
    def __init__(self, p, b=0, w=16):
        self.p = p
        self.b = b
        self.w = w
        
    def __call__(self, x):
        mask = torch.rand(x.shape) < (1-self.p)
        shaker = (self.w - self.b) * (torch.rand(x.shape) < 0.5) + self.b
        return x * mask + shaker * (~mask)


class Clamp:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        return torch.clamp(x, min=self.lower, max=self.upper)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return torch.div(x - self.mean, self.std)
