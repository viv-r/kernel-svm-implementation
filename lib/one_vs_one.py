import numpy as np
from lib.gradient_descent import gradient_descent
from itertools import combinations
from tqdm import tqdm
from lib.kernels.utils import kerneleval, computegram

def get_subset(options, dataset, class_a, class_b):
    """
    Returns a subset of the dataset that only contains the
    classes (class_a) and (class_b)
    """
    x_train, y_train = dataset
    train_idx = (y_train == class_a) ^ (y_train == class_b)

    subset_train_x = x_train[train_idx]
    subset_train_y = y_train[train_idx]

    # relabel to +/-1
    subset_train_y[subset_train_y == class_a] = -1
    subset_train_y[subset_train_y == class_b] = 1

    return subset_train_x, subset_train_y


def fit_single_class(options, dataset, class_a, class_b):
    """
    Fits a model for the given pair of classes: class_{a|b}
    """
    tx, ty = get_subset(options, dataset, class_a, class_b)

    K = computegram(tx, options['kernel'])
    b, history = gradient_descent(options, K, ty)

    return (b, history, tx)

def fit(options, dataset, classes):
    pairs = list(combinations(classes, 2))
    models = [fit_single_class(options, dataset, *p) for p in tqdm(pairs)]
    return {a: b for a, b in zip(pairs, models)}


def predict(options, models, x, classes, beta_version=None):
    """
    Predicts classes using the majority vote rule
    """
    votes = np.zeros((x.shape[0], len(classes)))
    for a, b in models:
        beta, history, tx = models[(a, b)]
        if beta_version is not None:
            idx = min(beta_version, len(history) - 1)
            beta = history[idx]

        yhat = kerneleval(tx, x, options['kernel']) @ beta > 0
        votes[:, np.argwhere(classes==b)[0][0]] += yhat
        votes[:, np.argwhere(classes==a)[0][0]] += 1 - yhat

    return classes[np.argmax(votes, axis=1)]
