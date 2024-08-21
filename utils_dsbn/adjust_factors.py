"""
Author:  Qi tong Chen
Date: 2024.08.20
Adaptation function
"""
import math
import matplotlib.pyplot as plt
from pylab import *


def amplification_factor(epoch, warm_epoch, max_epoch, gamma=10, rate=1):
    """
    Amplification Factor
    :param epoch: Current epoch
    :param warm_epoch: Pre-training steps
    :param max_epoch: Max epoch
    :param gamma: constant
    :return: [0~1]
    """
    if epoch > warm_epoch:
        p = (epoch - warm_epoch) / float(max_epoch)
        p = max(min(p, 1.0), 0.0)  # p denotes training progress linearly changing from o to 1
        den = 1.0 + math.exp(-gamma * p)
        lamb = (2.0 / den - 1.0)
    else:
        lamb = 0
    return min(lamb, 1.0) * rate


def attenuation_factor(epoch, warm_epoch, alpha=10, beta=0.75, max_epoch=2000):
    """
    Attenuation factor
    Args:
        epoch: Current epoch
        alpha: Constants
        beta: Constants
        max_epoch: Max epoch
        p represents the relative value of the iteration process, that is,
        the ratio of the current number of iterations to the total number of iterations
        p = float(i_iter) / num_steps, changing from 0 to 1

    Returns:

    """
    if epoch < warm_epoch:
        return 1
    else:
        p = (epoch - warm_epoch) / float(max_epoch)
        p = max(min(p, 1.0), 0.0)  # p denotes training progress linearly changing from o to 1, p[0~1]
        adjust_lr = 1 / (((1 + alpha * p) ** (beta)))

    return adjust_lr


if __name__ == '__main__':

    warm_epoch = 30
    max_epoch = 500
    adaptation_lamd = []
    current_lr = []
    for epoch in range(max_epoch):
        adaptation_lamda = amplification_factor(epoch=epoch, warm_epoch=warm_epoch, max_epoch=max_epoch, gamma=10, rate=1)
        adaptation_lamd.append(adaptation_lamda)

        adjust_lr = attenuation_factor(epoch=epoch, warm_epoch=warm_epoch, alpha=10, beta=0.75, max_epoch=max_epoch)
        current_lr.append(adjust_lr)
    plt.plot(adaptation_lamd, label='adaptation_lamd')
    plt.plot(current_lr, label='current_lr')
    plt.show()
