import torch.optim as optim


def get_optimizer(name, parameters, lr, momentum=None, weight_decay=0):
    """
    Returns the optimizer based on the provided name.

    Parameters:
    - name (str): Name of the optimizer ('SGD', 'Adam', 'Adagrad', ...)
    - parameters (iterable): Iterable of parameters to optimize.
    - lr (float): Learning rate.
    - momentum (float, optional): Momentum factor. Only used by some optimizers.
    - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.

    Returns:
    - Optimizer object
    """

    if name.lower() == 'sgd':
        optimizer=optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name.lower() == "adam":
        optimizer=optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name.lower() == "adagrad":
        optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{name}' not recognized.")

    return optimizer

