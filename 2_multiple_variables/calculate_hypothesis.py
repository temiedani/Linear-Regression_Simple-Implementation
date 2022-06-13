import numpy as np


def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0
    #########################################
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis = np.sum(np.dot(X[i], theta))
    # /
    #hypothesis = np.sum (np.array(X[i]) * np.array(theta))
    return hypothesis
