import os
import sys
import pickle
from dataset import OptDigits
from model import NonlinearNN
from trainer import Trainer
from transforms import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

datapath = "/Users/matthewjin/imajin/correlated-noise/optdigits"
savepath = "/Users/matthewjin/imajin/correlated-noise/checkpoints"

def run(logname, logdir, transform=None, epochs=100):
    train_path = os.path.join(datapath, "optdigits.tra")
    test_path = os.path.join(datapath, "optdigits.tes")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = OptDigits(train_path)
    test_dataset = OptDigits(test_path)

    trainer = Trainer(train_dataset, test_dataset, transform=transform, device=device)
    logs, models = [], []
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        model = NonlinearNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        trainer.resplit()
        trainer.reset_logger()
        log, model = trainer.train(model, optimizer, criterion, epochs=epochs, batch_size=32)
        logs.append(log)
        models.append(model)

    col = collect(logs)
    with open(os.path.join(savepath, logdir, logname + ".pkl"), 'wb') as f:
        pickle.dump(col, f)

def main():
    """Noise Variance"""
    # params = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3]
    # params = [1.6, 2.0, 2.5, 3.0, 4.0, 6.0]
    """Probabilities"""
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(params)):
        print(f"Run {i}")
        """Define augmentations"""
        # transform = None
        # transform = MultivariateNormalNoise(torch.zeros(64), params[i]*torch.eye(64))
        # transform = Mask(p=params[i])
        # transform = SparseNoise(m=1, p=params[i])
        transform = SaltAndPepper(p=params[i])
        """"""
        run(f"saltpepper_{params[i]}+1", "saltpepper", transform=transform, epochs=100)

if __name__ == "__main__":
    main()
