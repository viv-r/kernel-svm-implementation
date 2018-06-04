import numpy as np


def poly(X, Y, degree):
    return (X @ Y.T + 1)**degree