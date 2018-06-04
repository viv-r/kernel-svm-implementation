import numpy as np


def loss(K, y, alpha, L, h=0.5):
    _, n = K.shape

    z = 1 - y.reshape(-1,1) * K @ alpha
    z[(z < h) & (z > -h)] = ((z[(z < h) & (z > -h)] + h) ** 2) / (4 * h)
    z[z < -h] = 0

    return L * alpha.T @ K @ alpha + (1 / n) * z.sum()


def gradient(K, y, alpha, L, h=0.5):
    _, n = K.shape

    z = 1 - y.reshape(-1,1) * K @ alpha

    yK = -(y * K).T
    c = (z < h) & (z > -h)
    yK[c] = (yK[c].T * (z[c] + h) / (2 * h)).T
    yK[z < -h] = 0

    return 2 * L * K @ alpha + (1 / n) * yK.sum(axis=0)