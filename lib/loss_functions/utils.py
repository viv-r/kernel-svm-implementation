import lib.loss_functions.huber_hinge as huber_hinge
import lib.loss_functions.squared_hinge as squared_hinge


def get_loss_gradient_functions(objective):
    if objective == 'huber_hinge':
        return huber_hinge.loss, huber_hinge.gradient
    elif objective == 'squared_hinge':
        return squared_hinge.loss, squared_hinge.gradient

    raise Exception("Objective not found")