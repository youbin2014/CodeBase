import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Returns a learning rate scheduler.

    Parameters:
    - optimizer (torch.optim.Optimizer): Optimizer for which the scheduler will adjust the learning rate.
    - scheduler_type (str): Type of the scheduler. Supported values: ['StepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']
    - **kwargs: Arguments for the specific scheduler.

    Returns:
    - torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """

    if scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **kwargs)

    elif scheduler_type == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)

    elif scheduler_type == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    elif scheduler_type == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    else:
        raise ValueError(
            f"Scheduler type {scheduler_type} not recognized. Supported values are: ['StepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']")
