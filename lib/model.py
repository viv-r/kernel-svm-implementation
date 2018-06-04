
import numpy as np
import pandas as pd
import lib.one_vs_one as ovo
import lib.one_vs_rest as ovr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

default_options = dict(
    standardize = True,
    l2_lambda = 1,
    multiclass = 'ovo',
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

    def fit(self, x_train, y_train, x_val, y_val):
        if self.options['standardize']:
            self.scaler = StandardScaler()
            x_train = self.scaler.fit_transform(x_train)
            x_val = self.scaler.transform(x_val)

        self.dataset = (x_train, y_train, x_val, y_val)

        self.classes = np.unique(y_train)
        trainer = ovo if self.options['multiclass'] == 'ovo' else ovr
        self.models = trainer.fit(self.options, self.dataset, self.classes)

    def predict(self, x_test):
        if self.options['standardize']:
            x_test = self.scaler.transform(x_test)

        trainer = ovo if self.options['multiclass'] == 'ovo' else ovr
        return trainer.predict(self.options, self.models, x_test, self.classes)

    def plot(self, x, y, iters):
        if self.options['standardize']:
            x = self.scaler.transform(x)
        trainer = ovo if self.options['multiclass'] == 'ovo' else ovr
        scores = []
        for i in range(iters):
            yhat = trainer.predict(self.options, self.models, x, self.classes, i)
            sc = 1 - accuracy_score(yhat, y)
            scores.append(sc)

        pd.Series(scores).plot()