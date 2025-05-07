from types import MethodType
import re
import torch
from torch import nn


def set_requires_grad(model: nn.Module, freeze: int = 0, trainable_params: str = None):
    requires_grad = False if freeze else True
    for name, param in model.named_parameters():
        if not trainable_params or re.match(trainable_params, name):
            param.requires_grad = requires_grad
        else:
            param.requires_grad = False

    if freeze == 2:
        model.forward = torch.no_grad(model.forward)


def last_layer_as_adaptor(model: nn.Module):
    model_name = model.__class__.__name__
    if model_name == "CLIPVisionModel" or model_name == "SiglipVisionModel":
        last_layer = model.vision_model.encoder.layers[-1]
    elif model_name == "InternVisionModel":
        last_layer = model.encoder.layers[-1]
    elif model_name == "SamVisionEncoder":
        last_layer = model.neck
    else:
        raise NotImplementedError(f"Model type {type(model)} does not support last_layer_as_adaptor.")

    forward = last_layer.forward

    def xforward(self, *args, **kwargs):
        if self.training:
            torch.set_grad_enabled(True)
        output = forward(*args, **kwargs)

        return output

    last_layer.forward = MethodType(xforward, last_layer)
    for param in last_layer.parameters():
        param.requires_grad = True
