def get_subset(options, dataset, class_a):
    """
    Relabels the data to a One vs Rest scheme
    """
    x_train, y_train, x_val, y_val = dataset
    subset_train_y = y_train
    subset_val_y = y_val

    # relabel to +/-1
    subset_val_y[subset_val_y == class_a] = 1
    subset_val_y[subset_val_y != class_a] = -1
    subset_train_y[subset_train_y == class_a] = 1
    subset_train_y[subset_train_y != class_a] = -1

    return x_train, subset_train_y, x_val, subset_val_y


def fit_single_class(options, dataset, class_a):
    """
    Fits a ovr model for the given class
    """
    x_train, y_train, x_val, y_val = dataset
    tx, ty, vx, vy = get_subset(x_train, y_train, x_val, y_val, class_a)

    b, history = gradient_descent(tx, ty, L=L)
    return (b, history, None)
