import torch.nn as nn
import torch


class CustomDataParallel:
    def __init__(self, model, device_ids=None):
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.primary_device = torch.device(f"cuda:{self.device_ids[0]}")

        self.model = model
        if len(self.device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=self.device_ids)
        self.model = self.model.to(self.primary_device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        # This function allows flexibility to move models around like model.to(device)
        self.model = self.model.to(*args, **kwargs)
        return self

    def __getattr__(self, item):
        # This forwards any unimplemented calls to the internal model, be it the original model or DataParallel
        return getattr(self.model, item)

    def state_dict(self, *args, **kwargs):
        # Returns the state dict of the original model, not the DataParallel wrapper
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict(*args, **kwargs)
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        # Load state dict. Works with either the original model or DataParallel model
        return self.model.load_state_dict(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        # Returns the parameters of the original model, not the DataParallel wrapper
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.parameters(*args, **kwargs)
        else:
            return self.model.parameters(*args, **kwargs)


