import os
import sys
import pickle
from dataset import *
from model import NonlinearNN
from trainer import Trainer
from transforms import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

savepath = "/Users/matthewjin/imajin/correlated-noise/checkpoints"

def run(train_dataset, test_dataset, logname, logdir, in_dim=64,
              hidden_dim=32, transform=None, runs=10, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(train_dataset, test_dataset, transform=transform,
                      device=device)
    logs, models = [], []
    criterion = nn.MSELoss(reduction='sum') # nn.CrossEntropyLoss() 
    for i in range(runs):
        # model = NonlinearNN(in_dim=in_dim, hidden_dim=hidden_dim)
        # model.linear1.requires_grad_(False)
        model = nn.Linear(64, 10)
        """lr=0.001 for nonlinear, 0.0001 for linear"""
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        trainer.resplit()
        log, model = trainer.train(model, optimizer, criterion, epochs=epochs,
                                   batch_size=32)
        logs.append(log)
        models.append(model)

    col = collect(logs)
    with open(os.path.join(savepath, logdir, logname + ".pkl"), 'wb') as f:
        pickle.dump(col, f)

def run_optdigits(logname, logdir, params, runs=100, epochs=100):
    datapath = "/Users/matthewjin/imajin/correlated-noise/optdigits/"
    trainpath = os.path.join(datapath, "optdigits.tra")
    testpath = os.path.join(datapath, "optdigits.tes")
    train_dataset = OptDigits(trainpath)
    test_dataset = OptDigits(testpath)

    class_num, err = 8, 1e-6
    cov = train_dataset.data[train_dataset.classes==class_num].T.cov()
    eig, vec = torch.linalg.eig(cov)
    eig[eig==0] += err
    cov = vec.real @ torch.diag(eig.real) @ vec.T.real

    for i in range(len(params)):
        print(f"Run {i}: p={params[i]}")
        transform = None
        # transform = MultivariateNormalNoise(torch.zeros(64), params[i]*cov)
        run(train_dataset, test_dataset, logname=f"{logname}_{params[i]}",
            logdir=logdir, transform=transform, runs=runs, epochs=epochs,)
            # hidden_dim=params[i])

def run_mnist(logname, logdir, params, runs=10, epochs=250):
    datapath = "/Users/matthewjin/imajin/correlated-noise/mnist-back-image/"
    trainpath = os.path.join(datapath, "train.amat")
    testpath = os.path.join(datapath, "test.amat")
    train_dataset = MNIST(trainpath)
    test_dataset = MNIST(testpath)
    for i in range(len(params)):
        print(f"Run {i}: p={params[i]}")
        transform = None
        run(train_dataset, test_dataset, logname=f"{logname}_{params[i]}",
            logdir=logdir, transform=transform, runs=runs, epochs=epochs)


def main():
    params = [0]
    """Probabilities"""
    # params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    """Variances"""
    # params = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # params = [0.01, 0.05, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
    """Hidden dims"""
    # params = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    # params = [44, 48, 52, 56, 60, 64]
    # params = [5, 6, 7]

    run_optdigits(logname=f"opt_base_linear_ce", logdir=f".", params=params,
                  epochs=250, runs=25)

    ## Example augmentations
    """
    # transform = None
    # transform = MultivariateNormalNoise(torch.zeros(64), params[i]*torch.eye(64))
    # transform = MultivariateNormalNoise(torch.zeros(784),
    #                                     params[i]*torch.eye(784))
    # transform = Mask(p=params[i])
    # transform = SparseNoise(m=1, p=params[i])
    # transform = SaltAndPepper(p=params[i])
    # transform = ClassTransform(
    #         {9: MultivariateNormalNoise(torch.zeros(64),
    #                                     params[i]*torch.eye(64)),
    #         })
    """

if __name__ == "__main__":
    main()
