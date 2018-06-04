import lib.loss_functions.huber_hinge as huber_hinge
import lib.loss_functions.squared_hinge as squared_hinge


def get_loss_gradient_functions(objective):
    """
    Helper utility to return the (loss, gradient) function
    pair.
    """
    if objective == 'huber_hinge':
        return huber_hinge.loss, huber_hinge.gradient

    raise Exception("Objective not found")