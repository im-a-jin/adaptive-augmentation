import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class NonlinearNN(nn.Module):
    """Basic nonlinear classification model.

    The __init__ method takes in the number of input, hidden, and output
    dimensions.

    Args:
        in_dim (int): input dimensions
        hidden_dim (int): hidden dimensions
        out_dim (int): output dimensions

    Notes:
        out_dim=10 for the optdigits dataset
    """
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=10):
        super(NonlinearNN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def _init_weights(self, m):
        pass

    def forward(self, x):
        """Returns the result of the forward pass of the neural network."""
        return self.linear2(self.relu(self.linear1(x)))


class AdaptiveNoiseModel(nn.Module):
    """Model with adaptive noise component.
    """
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=10, num_classes=10,
                 aug_type='class'):
        super(AdaptiveNoiseModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.classes = [i for i in range(num_classes)]
        self.aug_type = aug_type
        if aug_type == 'global':
            self.cov = nn.Parameter(torch.ones(hidden_dim))
        elif aug_type == 'class':
            self.cov = nn.ParameterDict({str(c): nn.Parameter(torch.ones(hidden_dim))
                                         for c in self.classes})
        self.aug = MultivariateNormal(torch.zeros(hidden_dim),
                                      torch.eye(hidden_dim))

    def _add_noise(self, x, y):
        a = self.aug.expand(x[:, 0].size())
        n = a.sample(torch.Size([]))
        if self.aug_type == 'global':
            n = self.cov * a.sample(torch.Size([]))
            return x + n
        elif self.aug_type == 'class':
            l = []
            for c in self.classes:
                idx = (y == c)[:, None]
                l.append((x + self.cov[str(c)] * n) * idx)
            return sum(l)

    def forward(self, x, y, aug=True):
        # return self.linear2(self.relu(self.linear1(self._add_noise(x))))
        if aug:
            return self.linear2(self._add_noise(self.relu(self.linear1(x)), y))
        else:
            return self.linear2(self.relu(self.linear1(x)))


class RandomFeatureModel(nn.Module):
    def __init__(self, feature_dim, init_var, in_dim=64, out_dim=10, relu=True):
        super(RandomFeatureModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature_dim = feature_dim

        projection_matrix = torch.normal(torch.zeros(in_dim, feature_dim),
                                         torch.ones(in_dim, feature_dim) *
                                         np.sqrt(init_var))
        self.feature = nn.Parameter(projection_matrix)
        self.relu = nn.ReLU() if relu else lambda x: x
        self.linear = nn.Linear(feature_dim, out_dim)

    def features(self, x):
        return self.relu(x @ self.feature)

    def forward(self, x):
        return self.linear(self.relu(x @ self.feature))
