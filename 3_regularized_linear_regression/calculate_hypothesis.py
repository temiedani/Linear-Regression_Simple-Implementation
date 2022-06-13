import numpy as np


def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """

    #########################################
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis = X[i, 0] * theta[0] + X[i, 1] * theta[1] + (X[i, 1]**2) * theta[2] + (
        X[i, 1]**3) * theta[3] + (X[i, 1]**4) * theta[4] + (X[i, 1]**5) * theta[5]
    # /
    return hypothesis
