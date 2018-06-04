"""
The Model class.
"""

import numpy as np
import pandas as pd
import lib.one_vs_one as ovo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# default options that are used
# user specified options replace these.
default_options = dict(
    standardize = False,
    l2_lambda = 1,
    objective = 'huber_hinge',
    kernel = dict(
        fn = 'rbf',
        gamma = 0.1
    ),
    gradient_descent = dict(
        max_iter = 50,
        step_size = 0.001,
        eps = 0.001,
    ),
    backtrack = dict(
        max_iter = 20,
        alpha = 0.5,
        gamma = 0.8
    )
)

class Model():
    def __init__(self, options):
        self.options = {**default_options, **options}

    def fit(self, x_train, y_train):
        """
        Fits the model to the data provided
        """
        if self.options['standardize']:
            self.scaler = StandardScaler()
            x_train = self.scaler.fit_transform(x_train)

        self.dataset = (x_train, y_train)

        self.classes = np.unique(y_train)
        self.models = ovo.fit(self.options, self.dataset, self.classes)

    def predict(self, x_test):
        """
        Returns the model's predictions on x_test

        .fit(..) must be called prior to this.
        """
        if self.models is None:
            raise Exception("Fit was not called.")

        if self.options['standardize']:
            x_test = self.scaler.transform(x_test)

        return ovo.predict(self.options, self.models, x_test, self.classes)

    def plot(self, x, y, xt, yt, iters):
        """
        Plots the training and validation misclassfication
        error as a function of training iterations.
        """
        if self.options['standardize']:
            x = self.scaler.transform(x)
            xt = self.scaler.transform(xt)
        scores_x = []
        scores_xt = []
        for i in range(iters):
            yhat_x = ovo.predict(self.options, self.models, x, self.classes, i)
            scores_x.append(1 - accuracy_score(yhat_x, y))
            yhat_xt = ovo.predict(self.options, self.models, xt, self.classes, i)
            scores_xt.append(1 - accuracy_score(yhat_xt, yt))

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.set_title('Misclassfication plot')
        ax.plot(range(iters), scores_x, label='Training error')
        ax.plot(range(iters), scores_xt, label='Validation error')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Misclassfication (1-Accuracy)')
        ax.legend(loc='best')
