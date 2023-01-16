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

def get_nested_dict(d, *keys):
    """Gets the value corresponding to the keys from nested dict d."""
    for k in keys:
        d = d.__getitem__(k)
    return d

def map_nested_dicts(d, f):
    """Maps a function to all values in a nested dictionary."""
    if isinstance(d, dict):
        return {k: map_nested_dicts(v, f) for k, v in d.items()}
    else:
        return f(d)

def collect(logs):
    """Combines a list of logs into a single nested dictionary."""
    arr = {}
    for keys in sorted(logs[0].keylist, key=len):
        a = arr
        for k in keys[:-1]:
            if k not in a.keys():
                a[k] = {}
            a = a[k]
        a[keys[-1]] = np.empty((len(logs), len(logs[0][keys])))
        for i, l in enumerate(logs):
            a[keys[-1]][i] = np.array(l[keys])
    return arr


class ListLogger():
    """Logger class for getting, setting, and appending values to a nested
    dictionary. Uses list as the base datatype.
    """
    def __init__(self):
        self.dct = {}
        self.keylist = set()

    def __getitem__(self, keys):
        """Gets the value corresponding to the keys and returns None if the key
        list is not found.
        """
        if type(keys) is str:
            keys = tuple([keys])
        d = self.dct
        for k in keys:
            d = d[k]
        return d

    def __setitem__(self, keys, value):
        """Sets the key list in the nested dictionary to the provided value."""
        d = self.dct
        for k in keys[:-1]:
            if k not in d.keys():
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value if type(value) is list else [value]
        self.keylist.add(keys)

    def _append(self, keys, value):
        """Appends the value to the numpy array stored at the provided key
        list."""
        d = self.dct
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] += value if type(value) is list else [value]

    # TODO: add math functionality?
    def log(self, *keys, v=0.0):
        """Stores the value at the provided keys."""
        if keys in self.keylist:
            self._append(keys, v)
        else:
            self.__setitem__(keys, v)

    def __repr__(self):
        return str(self.dct)
