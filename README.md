
## Kernel SVM Implementation

This repository contains an implementation of a kernel SVM classifier.

### Requirements:

- numpy
- matplotlib

### Datasets:

The datasets used in the examples are taken from the following sources:

- Vowels dataset:  https://statweb.stanford.edu/~tibs/ElemStatLearn/
- Digits dataset: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

### Usage:

Instantiate the model:

```python
from lib import Model

options = { ... }
model = Model(options)
```

The fit and predict api is similar to sklearn's

```python
model.fit(x_train, y_train)
yhat = model.predict(x_test)
```

### Options

The model can be configured by passing a dictionary of options to the constructor.
The structure of the options object is as follows:

```typescript
{
    standardize: boolean, // (if the data should be standardized before fit)
    l2_lambda: float, // (the L2 regularization penalty)
    objective: 'huber_hinge' | 'squared_hinge', // (the loss function to use)
    kernel: {
        fn: 'rbf' | 'linear' | 'poly',  // (the kernel function to use)
        gamma: float // (only used if fn == 'rbf')
        degree: int // (only used if fn == 'poly')
    ),
    gradient_descent: {
        max_iter: int, // (the maximum iterations of the gradient descent algorithm)
        step_size: float, // (the initial step size)
        eps: float // (stop's training early if the norm of the gradient is smaller than this)
    ),
    backtrack: {
        max_iter: int, // (max iterations to find the step size using backtracking)
        alpha: float, // (backtracking control parameter)
        gamma: float // (step-size multiplier per iteration)
    }
}
```
