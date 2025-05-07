import re
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer, Linear as LoraLinear

from qqmm.utils.model_utils import load_state_dict_file


def make_lora(model, lora_param, dtype='auto', keep_requires_grad=False):
    requires_grad_params = []
    if keep_requires_grad:
        for param in model.parameters():
            if param.requires_grad:
                requires_grad_params.append(param)

    target_linear_modules = get_target_linear_modules(model, lora_param['target_modules'])

    peft_config = LoraConfig(
        target_modules=target_linear_modules,
        inference_mode=lora_param.get('inference_mode', False),
        r=lora_param['lora_r'],
        lora_alpha=lora_param['lora_alpha'],
        lora_dropout=lora_param['lora_dropout'],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    if dtype is not None:
        def as_dtype(m, dtype_):
            if dtype_ == torch.float16:
                m.half()
            elif dtype_ == torch.bfloat16:
                m.bfloat16()
            else:
                raise NotImplementedError(f"lora dtype conversion does not support {dtype_}")

        def to_dtype(m):
            if isinstance(m, LoraLayer):
                as_dtype(m, dtype)

        def auto_dtype(m):
            if isinstance(m, LoraLinear) and next(m.lora_A.parameters()).dtype != m.weight.dtype:
                as_dtype(m, m.weight.dtype)

        if dtype == 'auto':
            model.apply(auto_dtype)
        else:
            model.apply(to_dtype)

    if 'pretrained_path' in lora_param:
        device = next(model.parameters()).device
        load_state_dict_file(lora_param['pretrained_path'], model,
                             map_location=device, strict=False, ignore_unexpected_keys=False)
        print(f"Loaded LoRa: {lora_param['pretrained_path']}")

    for param in requires_grad_params:
        param.requires_grad = True

    model.print_trainable_parameters()

    if lora_param.get('merge_and_unload', False):
        # assert lora_param['inference_mode'], "lora must be in inference_mode when merge_and_unload."
        model = model.merge_and_unload()

    return model


def get_target_linear_modules(model, target_modules_re: str):
    target_modules = []
    for name, module in model.named_modules():
        if re.fullmatch(target_modules_re, name) and isinstance(module, nn.Linear):
            target_modules.append(name)

    return target_modules


def get_modules_to_save(model):
    modules_to_save = []

    for name, module in model.named_modules():
        requires_grad = False
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                requires_grad = True
                break

        if requires_grad:
            modules_to_save.append(name)

    return modules_to_save
