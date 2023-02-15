import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x, **kwargs):
        for transform in self.transforms:
            x = transform(x, **kwargs)
        return x


class MultivariateNormalNoise:
    def __init__(self, loc, covariance_matrix, f=lambda x: x):
        self.f = f
        self.m = MultivariateNormal(loc, covariance_matrix)

    def __call__(self, x, **kwargs):
        # https://bochang.me/blog/posts/pytorch-distributions/#row-4
        n = self.m.expand(x[:, 0].size())
        noise = self.f(n.sample(torch.Size([])))
        return x + noise


class Mask:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, **kwargs):
        mask = torch.rand(x.shape) < (1 - self.p)
        return x * mask


class SparseNoise:
    def __init__(self, p=0.5, m=1):
        self.p = p
        self.m = m
        
    def __call__(self, x, **kwargs):
        mask = torch.rand(x.shape) < self.p
        return x + self.m * mask


class SaltAndPepper:
    def __init__(self, p=0.5, b=0, w=16):
        self.p = p
        self.b = b
        self.w = w
        
    def __call__(self, x, **kwargs):
        mask = torch.rand(x.shape) < (1-self.p)
        shaker = (self.w - self.b) * (torch.rand(x.shape) < 0.5) + self.b
        return x * mask + shaker * (~mask)


class Clamp:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, x, **kwargs):
        return torch.clamp(x, min=self.lower, max=self.upper)


class MinMaxNormalization:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, x, **kwargs):
        return torch.div(x - self.lo, self.hi - self.lo)


class ClassTransform:
    def __init__(self, transforms):
        self.transforms = transforms
        # self.transforms = {
        #         0: Mask(0.5),
        #         1: SparseNoise(0.5),
        #         2: lambda x: x,
        #         3: MultivariateNormalNoise(torch.zeros(64), torch.eye(64)),
        #         # SparseNoise(0.5),
        #         4: MultivariateNormalNoise(torch.zeros(64), torch.eye(64)),
        #         5: SparseNoise(0.5),
        #         6: SparseNoise(0.5),
        #         7: MultivariateNormalNoise(torch.zeros(64), torch.eye(64)),
        #         # lambda x, **kwargs: x, # SaltAndPepper(0.5),
        #         8: MultivariateNormalNoise(torch.zeros(64), torch.eye(64)),
        #         # lambda x, **kwargs: x, # Mask(0.5),
        #         9: SparseNoise(0.5),
        #         }

    def _apply_transforms(self, x, y):
        if y in self.transforms.keys():
            transform = self.transforms[y]
        else:
            transform = lambda x: x
        return transform(x)

    def __call__(self, x, y=None):
        x_ = torch.empty(x.shape)
        for c in range(10):
            idx = y == c
            x_[idx] = self._apply_transforms(x[idx], c)
        return x_

