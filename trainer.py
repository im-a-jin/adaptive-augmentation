import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import utils

class Trainer():
    """General class for training models.

    The __init__ method takes in the model and the training dataset, as well as
    any transforms to apply to the dataset.

    Args:
        model: PyTorch neural network module
        dataset: Training dataset
        transform: Transforms to apply to the input dataset
        device: Device to run calculations on

    Attributes:
        train_split: Training split (80% of dataset)
        val_split: Validation split (20% of dataset)
        log: Dictionary of metrics to log while training
    """
    # TODO: Add model initialization and reset
    def __init__(self, model, dataset, transform=None, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.transform = transform
        self.device = device

        self.reset_splits()
        self.reset_model()
        self.reset_log()

    def reset_model(self):
        """Resets model parameters."""
        self.model.apply(utils.reset_parameters)

    def reset_log(self):
        """Resets log."""
        self.log = {
                'epochs': [], 
                'train_loss': [], 'train_acc': [], 
                'val_loss': [], 'val_acc': [],
                'train_class_loss': {}, 'train_class_acc': {},
                'val_class_loss': {}, 'val_class_acc': {},
                }
        for c in torch.unique(self.dataset.classes):
            c = c.item()
            self.log['train_class_loss'][c] = []
            self.log['train_class_acc'][c] = []
            self.log['val_class_loss'][c] = []
            self.log['val_class_acc'][c] = []

    def reset_splits(self):
        """Resets dataset splits."""
        self.train_split, self.val_split = random_split(self.dataset, [0.8, 0.2])

    def train(self, optimizer, criterion, epochs=1, batch_size=1, leave=False):
        """Training loop using the provided optimizer, criterion, and training
        parameters.
        """
        train_loader = DataLoader(self.train_split, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in tqdm(range(1, epochs+1), leave=leave):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                if self.transform is not None:
                    X = self.transform(X)
                optimizer.zero_grad()
                logits = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            train_loss, train_acc, tc_loss, tc_acc = self.eval(self.train_split, criterion)
            val_loss, val_acc, vc_loss, vc_acc = self.eval(self.val_split, criterion)

            # TODO: modularize logging step?
            self.log['epochs'].append(epoch)
            self.log['train_loss'].append(train_loss)
            self.log['train_acc'].append(train_acc)
            self.log['val_loss'].append(val_loss)
            self.log['val_acc'].append(val_acc)
            for c in tc_loss.keys():
                self.log['train_class_loss'][c].append(tc_loss[c])
                self.log['train_class_acc'][c].append(tc_acc[c])
                self.log['val_class_loss'][c].append(vc_loss[c])
                self.log['val_class_acc'][c].append(vc_acc[c])
        utils.map_nested_dicts(np.array, self.log)
        return self.log

    def eval(self, dataset, criterion):
        """Evaluation loop using the provided criterion.
        """
        loader = DataLoader(dataset, batch_size=len(dataset))
        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = criterion(logits, y)
                y_hat = torch.argmax(logits, dim=1)
                y_ = (y_hat == y)
                acc = torch.sum(y_) / len(y)
                # Class-specific logging
                class_loss, class_acc = {}, {}
                for c in torch.unique(y):
                    c = c.item()
                    cargs = torch.argwhere(y == c).squeeze() # Squeeze shape [c, 1] to [c] for indexing
                    class_loss[c] = criterion(logits[cargs, :], y[cargs])
                    class_acc[c] = torch.sum(y_hat[cargs] == y[cargs]) / len(y[cargs])
        return loss, acc, class_loss, class_acc
