import numpy as np


def poly(X, Y, degree):
    """
    Polynomial kernel implementation.

    X, Y: The matrices between which pairwise distances
    are computed.

    degree: The degree of the polynomial
    """
    return (X @ Y.T + 1)**degree