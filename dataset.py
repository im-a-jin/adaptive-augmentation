import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class OptDigits(Dataset):
    """Dataset class for the optdigits dataset.

    The __init__ method takes in the filepath to a data file and parses them
    according to the convention described in optdigits.names

    Args:
        filepath (str): path to the data file

    Attributes:
        data (list of list of int): List of groups of 64 digits representing the
        input image
        classes (list of int): List of classes
    """
    def __init__(self, filepath):
        self.data = []
        self.classes = []
        with open(filepath) as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(np.array(list(map(int, row[:-1]))))
                self.classes.append(int(row[-1]))
        self.data = torch.FloatTensor(np.array(self.data))      # Inputs need to be float
        self.classes = torch.LongTensor(np.array(self.classes)) # Targets need to be long

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.classes)

    def __getitem__(self, index):
        """Returns the image data and class of the sample at the provided index"""
        return self.data[index], self.classes[index]

    def classlist(self):
        return self.classes.tolist()

class MNIST(Dataset):
    def __init__(self, filepath):
        amat = np.loadtxt(filepath)
        self.data = torch.FloatTensor(amat[:, :-1])
        self.classes = torch.LongTensor(amat[:, -1])

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        return self.data[index], self.classes[index]
