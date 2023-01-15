"""
This utility module is built specifically for interacting with the optdigits
dataset. Functions within assume properties specific to the dataset that may not
apply elsewhere.
"""
import numpy as np

def toImage(vals):
    """Converts list of 64 ints to a matrix of 8 by 8 ints."""
    return np.reshape(vals, (8, 8))

def toList(vals):
    """Converts 8 by 8 int matrix to list of 64 ints."""
    return np.reshape(vals, (64))

def reset_parameters(m):
    """Resets the parameters of a module using module.apply(reset_parameters)."""
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()

def map_nested_dicts(f, d):
    """Maps a function to all values in a nested dictionary."""
    for k, v in d.items():
        if isinstance(v, dict):
            map_nested_dicts(f, v)
        else:
            d[k] = f(v)
