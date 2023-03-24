import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x, **kwargs):
        for transform in self.transforms:
            x = transform(x, **kwargs)
        return x

    def log(self):
        raise NotImplementedError


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


class StochasticNoiseAugmentation:
    """
    s: data-independent covariance variance
    d: covariance d
    C: list of class ids
    l: discount factor
    k: number of minibatches to average over
    bs: batch size
    """
    def __init__(self, s, d, C, l, k, bs):
        self.alpha = MultivariateNormal(torch.ones(d), s*torch.eye(d))
        self.beta = MultivariateNormal(torch.zeros(d), s*torch.eye(d))
        self.s = s
        self.d = d
        self.C = C
        self.l = l
        self.k = k
        self.count = 0
        self.augs = {c.item(): None for c in C}
        self.covs = {c: torch.zeros(d,d) for c in C}
        self.global_cov = torch.zeros(d,d)
        self.x = torch.zeros(bs*k,d)
        self.y = torch.zeros(bs*k)

    def _make_psd(self, cov):
        eig = torch.linalg.eigvalsh(cov)
        err = eig.real.min()
        return cov + (abs(err)+1e-6) * torch.eye(cov.size(0))

    def _store_cov(self, x, y):
        idx, bs = self.count*self.k, y.size(0)
        self.x[idx:idx+bs] = x
        self.y[idx:idx+bs] = y

    def _update_cov(self):
        cov = self.x.T.cov()
        # self.global_cov = self.l * self.global_cov + (1 - self.l) * cov
        # global_eig, _ = torch.linalg.eig(self.global_cov)
        for c in self.C:
            idx = self.y == c
            cov = self.x[idx].T.cov()
            self.covs[c] = self.l * self.covs[c] + (1 - self.l) * cov
            # eig, _ = torch.linalg.eig(self.covs[c])
            r = 1
            if c == 0 or c == 1:
                r = 0.1
            try:
                self.augs[c] = MultivariateNormal(torch.zeros(self.d), self.covs[c])
            except ValueError:
                self.covs[c] = self._make_psd(self.covs[c])
                self.augs[c] = MultivariateNormal(torch.zeros(self.d), self.covs[c])

    def _apply_transform(self, x, c):
        s = torch.Size([])
        a = self.alpha.expand(x[:, 0].size())
        b = self.beta.expand(x[:, 0].size())
        x_ = x # a.sample(s) * x + b.sample(s)
        if self.augs[c] is not None:
            eps = self.augs[c].expand(x[:, 0].size())
            x_ += eps.sample(s)
        return x_

    def __call__(self, x, y=None, **kwargs):
        if self.count != self.k:
            self._store_cov(x, y)
            self.count += 1
        else:
            self._update_cov()
            self.count = 0
        x_ = torch.empty(x.shape)
        for c in self.C:
            idx = y == c
            x_[idx] = self._apply_transform(x[idx], c.item())
        return x_

    def log(self):
        return self.covs


class AdaptiveNoiseAugmentation:
    def __init__(self, d):
        self.d = d
        self.cov = nn.Parameter(torch.ones(d))
        self.aug = MultivariateNormal(torch.zeros(d), torch.eye(d))

    def __call__(self, x, **kwargs):
        a = self.aug.expand(x[:, 0].size())
        n = self.cov * self.aug.sample(torch.Size([]))
        return x + n

    def log(self):
        return self.cov


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

    def __call__(self, x, y=None, **kwargs):
        x_ = torch.empty(x.shape)
        for c in range(10):
            idx = y == c
            x_[idx] = self._apply_transforms(x[idx], c)
        return x_

