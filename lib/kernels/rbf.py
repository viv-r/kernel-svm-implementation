import numpy as np


def rbf(X, Y, gamma):
    norm = np.linalg.norm
    dist = (norm(X, axis=1)**2)[:,None] + (norm(Y, axis=1)**2)[None,:] - 2*X@Y.T
    return np.exp(-gamma * dist)
