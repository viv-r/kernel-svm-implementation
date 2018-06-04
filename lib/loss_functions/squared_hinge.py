import numpy as np


def loss(x, y, beta, L, h=0.5):
    d, n = x.shape

    z = 1 - y.reshape(-1,1) * x.T @ beta
    z[(z < h) & (z > -h)] = ((z[(z < h) & (z > -h)] + h) ** 2) / (4 * h)
    z[z < -h] = 0

    return L * np.linalg.norm(beta) ** 2 + (1 / n) * z.sum()


def gradient(x, y, beta, L, h=0.5):
    d, n = x.shape

    z = 1 - y.reshape(-1,1) * x.T @ beta

    yx = (-y * x).T
    c = (z < h) & (z > -h)
    yx[c] = (yx[c].T * (z[c] + h) / (2 * h)).T
    yx[z < -h] = 0

    return 2 * L * beta + (1 / n) * yx.sum(axis=0)