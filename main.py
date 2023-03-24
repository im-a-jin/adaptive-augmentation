import os
import sys
import pickle
from dataset import *
from model import *
from trainer import Trainer
from transforms import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

savepath = "/Users/matthewjin/imajin/correlated-noise/checkpoints"

def run_rfa(feature_dim, init_var, relu, train_dataset, test_dataset, transform,
            logname, logdir, runs=100):
    trainer = Trainer(train_dataset, test_dataset)
    logs, models = [], []
    criterion = nn.MSELoss(reduction='sum')
    X = torch.clone(train_dataset.data)
    y = F.one_hot(train_dataset.classes, num_classes=10)
    for i in range(runs):
        model = RandomFeatureModel(feature_dim, init_var, relu=relu)
        fX = model.features(transform(X))
        fX_b = torch.ones(fX.size(0), fX.size(1)+1)
        fX_b[:, :fX.size(1)] = fX
        theta = torch.linalg.pinv(fX_b.T @ fX_b) @ fX_b.T @ y.float()
        model.linear.weight = nn.Parameter(theta[:feature_dim, :].T,
                                           requires_grad=False)
        model.linear.bias = nn.Parameter(theta[feature_dim, :],
                                         requires_grad=False)
        """Logging"""
        logger = ListLogger()
        l0, a0, cl0, ca0 = trainer.eval(model, train_dataset, criterion)
        l1, a1, cl1, ca1 = trainer.eval(model, test_dataset, criterion)
        logger.log('train_loss', v=l0)
        logger.log('train_acc', v=a0)
        logger.log('test_loss', v=l1)
        logger.log('test_acc', v=a1)
        for c in range(10):
            logger.log('train_class_loss', c, v=cl0[c])
            logger.log('train_class_acc', c, v=ca0[c])
            logger.log('test_class_loss', c, v=cl1[c])
            logger.log('test_class_acc', c, v=ca1[c])
        logs.append(logger)
        models.append(model.feature.data)
        """"""

    logs = collect(logs)
    with open(os.path.join(savepath, logdir, logname + "_logs.pkl"), 'wb') as f:
        pickle.dump(logs, f)
    with open(os.path.join(savepath, logdir, logname + "_models.pkl"), 'wb') as f:
        pickle.dump(models, f)


def run_gradient(train_dataset, test_dataset, logname, logdir, in_dim=64,
                 hidden_dim=32, transform=None, aug_type='global', runs=10,
                 epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(train_dataset, test_dataset, transform=transform,
                      device=device)
    logs, models = [], []
    criterion = nn.CrossEntropyLoss() # nn.MSELoss(reduction='sum')
    for i in range(runs):
        # model = NonlinearNN(in_dim=in_dim, hidden_dim=hidden_dim)
        # model.linear1.requires_grad_(False)
        # model = nn.Linear(64, 10)
        model = AdaptiveNoiseModel(in_dim=in_dim, hidden_dim=hidden_dim,
                                   aug_type=aug_type)
        """lr=0.01 for nonlinear, 0.0001 for linear"""
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        trainer.resplit()
        log, model = trainer.train(model, optimizer, criterion, epochs=epochs,
                                   batch_size=32)
        logs.append(log)
        models.append(model)

    col = collect(logs)
    with open(os.path.join(savepath, logdir, logname + "_logs.pkl"), 'wb') as f:
        pickle.dump(col, f)
    with open(os.path.join(savepath, logdir, logname + "_models.pkl"), 'wb') as f:
        pickle.dump(models, f)

    # if transform is not None:
    #     with open(os.path.join(savepath, logdir, logname + "_augs.pkl"), 'wb') as f:
    #         pickle.dump(transform.log(), f)


def run_optdigits(logname, logdir, params, aug_type, runs=100, epochs=100):
    datapath = "/Users/matthewjin/imajin/correlated-noise/optdigits/"
    trainpath = os.path.join(datapath, "optdigits.tra")
    testpath = os.path.join(datapath, "optdigits.tes")
    train_dataset = OptDigits(trainpath)
    test_dataset = OptDigits(testpath)

    classes = np.unique(train_dataset.classes)

    for i in range(len(params)):
        print(f"Run {i}: p={params[i]}")
        transform = None
        # transform = MinMaxNormalization(0, 16)
        # transform = MultivariateNormalNoise(torch.zeros(64),
        #                                     params[i]*torch.eye(64))
        # transform = StochasticNoiseAugmentation(s=1.0, d=64, C=classes, l=0.3,
        #                                         k=32, bs=32)
        # transform = AdaptiveNoiseAugmentation(d=64)

        run_gradient(train_dataset, test_dataset,
                     logname=f"{logname}_{params[i]}", logdir=logdir,
                     transform=transform, aug_type=aug_type, runs=runs,
                     epochs=epochs,)

        # # Feature dim sweep
        # run_rfa(params[i], 1, True, train_dataset, test_dataset,
        #         transform=transform, logname=f"{logname}_{params[i]}",
        #         logdir=logdir, runs=runs)

        # # Augmentation sweep (fdim=128)
        # run_rfa(128, 1, True, train_dataset, test_dataset, transform=transform,
        #         logname=f"{logname}_{params[i]}", logdir=logdir, runs=runs)


def run_mnist(logname, logdir, params, aug_type, runs=10, epochs=250):
    datapath = "/Users/matthewjin/imajin/correlated-noise/mnist-back-image/"
    trainpath = os.path.join(datapath, "train.amat")
    testpath = os.path.join(datapath, "test.amat")
    train_dataset = MNIST(trainpath)
    test_dataset = MNIST(testpath)
    classes = np.unique(train_dataset.classes)

    for i in range(len(params)):
        print(f"Run {i}: {logname}, p={params[i]}")
        transform=None
        # transform = StochasticNoiseAugmentation(s=1.0, d=784, C=classes, l=0.3,
        #                                         k=32, bs=32)
        run_gradient(train_dataset, test_dataset,
                     logname=f"{logname}_{params[i]}", logdir=logdir,
                     in_dim=784, hidden_dim=392, transform=transform,
                     aug_type=aug_type, runs=runs, epochs=epochs)

def main():
    params = ['latent']
    """Probabilities"""
    # params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    """Variances"""
    # params = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # params = [0.01, 0.05, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
    """Hidden dims"""
    # params = [8, 16, 32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    #           448, 512]

    # TODO: Ask about normalizing or not?
    # run_optdigits(logname=f"class_500_newloss",
    #               logdir=f"optdigits/2layer_adaptive_aug",
    #               aug_type = 'class', params=params, epochs=500, runs=25)
    run_mnist(logname=f"class", logdir=f"mnist/2layer_adaptive_aug",
              aug_type='class', params=params, epochs=250, runs=10)

if __name__ == "__main__":
    main()
