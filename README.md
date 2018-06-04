
### Kernel SVM Implementation

This repositiory contains an implementation of an kernel SVM classifier.

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
    standardize: boolean,
    l2_lambda: float,
    objective: 'huber_hinge' | 'squared_hinge',
    kernel = {
        fn = 'rbf' | 'linear' | 'poly',
        gamma: float (only used if fn == 'rbf')
        degree: int (only used if fn == 'poly')
    ),
    gradient_descent: {
        max_iter: int,
        step_size: float,
        eps: float
    ),
    backtrack: {
        max_iter: int,
        alpha: float,
        gamma: float
    }
}
```