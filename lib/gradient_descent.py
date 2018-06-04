"""
Implementation of the backtracking line search algorithm
and the the fast gradient descent algorithm.
"""
import numpy as np
from lib.loss_functions.utils import get_loss_gradient_functions

def backtrack(loss, K, y, beta, L, eta, grad, norm, alpha, gamma, max_iter):
    """
    Implements the backtracking line search algorithm
    """
    for _ in range(max_iter):
        a = loss(K, y, beta - eta * grad, L)
        b = loss(K, y, beta, L) - alpha * eta * norm

        if a <= b:
            return eta
        eta *= gamma

    return eta


def gradient_descent(options, K, y):
    """
    Implements the fast gradient descent algorithm
    """
    L = options['l2_lambda']
    max_iter = options['gradient_descent']['max_iter']
    eta = options['gradient_descent']['step_size']
    eps = options['gradient_descent']['eps']
    alpha = options['backtrack']['alpha']
    gamma = options['backtrack']['gamma']
    bt_max_iter = options['backtrack']['max_iter']
    loss, gradient = get_loss_gradient_functions(options['objective'])

    d, n = K.shape
    theta = np.zeros(d)
    beta = np.zeros(d)
    history = [beta.copy()]

    for t in range(max_iter):
        grad = gradient(K, y, theta, L)
        norm_grad = np.linalg.norm(grad) ** 2

        if norm_grad < eps:
            break

        eta = backtrack(loss, K, y, theta, L, eta, grad, norm_grad, alpha, gamma, bt_max_iter)
        beta_new = theta - eta * grad
        theta = beta_new + ((t + 1) / (t + 4)) * (beta_new - beta)
        beta = beta_new
        history.append(beta.copy())

    return beta, history