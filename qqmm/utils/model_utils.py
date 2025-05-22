from typing import Optional, Sequence
from collections import OrderedDict
import glob
import re
import gc
import torch
from qqmm.utils.loggings import logger

BASIC_TYPES = (int, float, str, bool)


def to_device(entry, device: Optional[torch.device] = None, dtype: torch.dtype = None):
    if isinstance(entry, torch.Tensor):
        if not entry.is_floating_point() or entry.dtype == dtype:
            dtype = None
        if device is None and dtype is None:
            return entry
        return entry.to(device=device, dtype=dtype)
    elif isinstance(entry, Sequence):
        if all(isinstance(x, BASIC_TYPES) for x in entry):
            return entry
        return type(entry)(x if isinstance(x, BASIC_TYPES) else to_device(x, device, dtype) for x in entry)
    elif isinstance(entry, dict):
        return {k: to_device(v, device, dtype) for k, v in entry.items()}
    else:
        return entry


def load_state_dict_file(state_dict_path, model=None, map_location=None, strict=True,
                         ignore_missing_keys=None, ignore_unexpected_keys=None, prefix='default'):
    state_dict, missing_keys, unexpected_keys = None, None, None

    if isinstance(state_dict_path, list):
        state_dict, missing_keys, unexpected_keys = _load_state_dict_files(state_dict_path, model, map_location, prefix)
    elif not isinstance(state_dict_path, str):
        raise ValueError('state_dict_path should be a string or a list of strings, '
                         f'rather than type {type(state_dict_path)}.')
    elif glob.has_magic(state_dict_path):
        paths = glob.glob(state_dict_path)
        state_dict, missing_keys, unexpected_keys = _load_state_dict_files(paths, model, map_location, prefix)
    else:
        logger.info(f'Loading state_dict from {state_dict_path}')

        if model is None:
            map_location_, mmap = map_location, None
        else:
            map_location_, mmap = 'cpu', True

        state_dict = torch.load(state_dict_path, map_location=map_location_, mmap=mmap)

        if 'ds_config' in state_dict:
            state_dict = state_dict['module']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = _parse_state_dict(state_dict, prefix)
        if len(state_dict) == 0:
            raise RuntimeError(f'Loading empty state_dict: {state_dict_path}')

    if model is None:
        return state_dict
    else:
        if state_dict is not None:
            missing_keys, unexpected_keys = _load_state_dict(model, state_dict)
            # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            del state_dict
            torch.cuda.empty_cache()
            gc.collect()

        if ignore_missing_keys is None:
            ignore_missing_keys = not strict
        if ignore_unexpected_keys is None:
            ignore_unexpected_keys = not strict

        if not ignore_unexpected_keys and unexpected_keys:
            raise RuntimeError('Unexpected key(s) in state_dict: {}. '
                               .format(', '.join(f'"{k}"' for k in unexpected_keys)))
        if not ignore_missing_keys and missing_keys:
            raise RuntimeError('Missing key(s) in state_dict: {}. '
                               .format(', '.join(f'"{k}"' for k in missing_keys)))

        return missing_keys, unexpected_keys


def _load_state_dict(model, state_dict):
    if 'PeftModel' in [t.__name__ for t in type(model).__mro__]:
        adaptor_name = model.active_adapter
        state_dict_ = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace('.lora_A.weight', f'.lora_A.{adaptor_name}.weight') \
                 .replace('.lora_B.weight', f'.lora_B.{adaptor_name}.weight')
            if not k.startswith('base_model.model.'):
                k = 'base_model.model.' + k
            state_dict_[k] = v
        state_dict = state_dict_

    device_map = {n: p.device for n, p in model.named_parameters()}

    missing_keys, unexpected_keys = set(device_map.keys()), set()
    for n in list(state_dict.keys()):
        p = state_dict[n]
        if n not in device_map:
            unexpected_keys.add(n)
            continue
        device = device_map[n]
        param = {n: p.to(device)}
        missing_keys_, unexpected_keys_ = model.load_state_dict(param, strict=False)
        del param
        missing_keys = missing_keys & set(missing_keys_)
        unexpected_keys = unexpected_keys | set(unexpected_keys_)

    return missing_keys, unexpected_keys


def _load_state_dict_files(state_dict_paths, model=None, map_location=None, prefix='default'):
    state_dict, missing_keys, unexpected_keys = None, None, None

    rets = []
    for p in state_dict_paths:
        ret = load_state_dict_file(p, model,
                                   map_location=map_location,
                                   strict=False,
                                   ignore_missing_keys=True,
                                   ignore_unexpected_keys=True,
                                   prefix=prefix)
        rets.append(ret)

    if model is None:
        state_dict = rets[0]
        for sd in rets[1:]:
            state_dict.update(sd)
    else:
        missing_keys = list(set.intersection(*(set(ret[0]) for ret in rets)))
        unexpected_keys = list(set.union(*(set(ret[1]) for ret in rets)))

    return state_dict, missing_keys, unexpected_keys


def _parse_state_dict(state_dict, prefix='default'):
    if not prefix:
        return state_dict
    elif prefix == 'default':
        pattern = r'^(module\.)*(loss\.)?'
    else:
        pattern = prefix
        if not pattern.startswith('^'):
            pattern = '^' + pattern

    names = state_dict.keys()
    matches = [re.search(pattern, n) for n in names]
    matches = [m.group() if m else m for n, m in zip(names, matches)]

    if prefix is None:
        if any(m is None for m in matches):
            return state_dict

        min_len = min(len(m.split('.')) for m in matches)
        matches = ['.'.join(m.split('.')[:min_len-1]+['']) for m in matches]
        matches = [None if m.endswith('loss.') else m for m in matches]

    state_dict_ = {}
    for (k, v), m in zip(state_dict.items(), matches):
        if m is None:
            continue
        k_ = k[len(m):]
        state_dict_[k_] = v

    return state_dict_
