import torch.nn as nn


class Criterion:
    @staticmethod
    def get_criterion(name):
        """
        Returns the loss function (criterion) based on the provided name.

        Parameters:
        - name (str): Name of the loss function ('crossentropy', 'mse', ...)

        Returns:
        - Loss function (criterion)
        """
        criteria = {
            "crossentropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            # ... add other criteria as required
        }

        if name not in criteria:
            raise ValueError(f"Criterion '{name}' not recognized. Available options: {', '.join(criteria.keys())}")

        return criteria[name]
