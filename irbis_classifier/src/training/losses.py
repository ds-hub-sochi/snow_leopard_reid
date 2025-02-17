import inspect
import sys

import torch


class LossFactory:
    @staticmethod
    def get_loss(
        loss_name: str,        
    ) -> type[torch.nn.Module]:
        if hasattr(torch.nn, loss_name):
            return getattr(
                torch.nn,
                loss_name,
            )
        else:
            for object_name, object in inspect.getmembers(sys.modules[__name__]):
                if inspect.isclass(object) and object_name == loss_name:
                    return object
        
        raise ValueError(f"loss you've provided '{loss_name}' wasn't found")
