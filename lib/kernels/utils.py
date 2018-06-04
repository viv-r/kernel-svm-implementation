from lib.kernels.rbf import rbf
from lib.kernels.linear import linear
from lib.kernels.poly import poly


def kerneleval(X_test, X_train, kernel):
    """
    This function computes the pariwise distances between
    each row in X_test and X_train using the kernel
    specified in 'kernel'

    X_test, X_train: 2d np.arrays
    kernel: kernel parameters
    """
    if kernel is None:
        return X_train

    fn = kernel['fn']
    if fn == 'rbf':
        return rbf(X_train, X_test, gamma=kernel['gamma'])
    elif fn == 'poly':
        return poly(X_train, X_test, degree=kernel['degree'])
    elif fn == 'linear':
        return linear(X_train, X_test)


def computegram(X, kernel):
    """
    Computes the pariwise distances between each row of
    X, using the kernel specified in 'kernel'
    """
    if kernel is None:
        return X.T

    return kerneleval(X, X, kernel)