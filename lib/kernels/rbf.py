import numpy as np


def rbf(X, Y, gamma):
    """
    Polynomial kernel implementation.

    X, Y: The matrices between which pairwise distances
    are computed.

    gamma: Inverse of the influence of each training sample
    """
    norm = np.linalg.norm
    dist = (norm(X, axis=1)**2)[:,None] + (norm(Y, axis=1)**2)[None,:] - 2*X@Y.T
    return np.exp(-gamma * dist)
