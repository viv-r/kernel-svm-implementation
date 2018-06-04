import numpy as np


def linear(X, Y):
    """
    Linear kernel implementation.

    X, Y: The matrices between which pairwise distances
    are computed.
    """
    return X @ Y.T