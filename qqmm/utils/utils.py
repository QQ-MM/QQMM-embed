import os
import yaml
from .parameter_manage import Parameters


def load_config(name_or_path: str) -> Parameters:
    """
    Load QQMM config from a checkpoint or a config file.
    Args:
        name_or_path (str): Path to a checkpoint or a config file.
    """
    if not name_or_path.startswith('/'):
        if not name_or_path.endswith('.yaml'):
            if os.path.splitext(name_or_path)[1] != '':
                raise ValueError(f"name_or_path must be a yaml file, but got {os.path.splitext(name_or_path)[1]}")
            name_or_path += '.yaml'
        import qqmm
        name_or_path = os.path.abspath(os.path.join(qqmm.__file__, '..', 'configs', name_or_path))
    elif name_or_path.endswith('.yaml'):
        pass
    elif os.path.isdir(name_or_path):
        name_or_path = os.path.join(name_or_path, 'config.yaml')
    else:
        raise ValueError(f'Invalid name_or_path: {name_or_path}')

    config = Parameters()
    with open(name_or_path) as f:
        is_full_config = '_PARAMETERS_' in yaml.load(f, Loader=yaml.FullLoader)
    if not is_full_config:
        from qqmm import base_config
        config.merge_from_py(base_config)
    config.merge_from_yaml(name_or_path)

    return config
