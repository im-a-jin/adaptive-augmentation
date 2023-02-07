import numpy as np
import torch
import torch.nn as nn

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
