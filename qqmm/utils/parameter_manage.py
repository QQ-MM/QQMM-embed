import os
import argparse
import copy
import json
import yaml


class Parameters:
    PARAMETERS_KEY = '_PARAMETERS_'
    REMOVE_KEY = '_REMOVE_'
    BASE_KEY = '_BASE_'
    OVERWRITE_KEY = '_OVERWRITE_'

    def merge_from_args(self, args: argparse.Namespace, arg_prefix: str = 'params.', named_key: str = None, **kwargs):
        """
        Args:
            args (argparse.Namespace): return from argparse.ArgumentParser.parse_args, used to parse agrs from cmd.
            arg_prefix (str): prefix of arg to indicate parameter options, must end with '.', or equals ''.
            named_key (str): named_key to wrap the parameters to be merged.
            override (bool): override the existed value if True, otherwise skip.
        """
        assert arg_prefix.endswith('.') or len(arg_prefix) == 0, "arg_prefix must end with '.', or equals ''."
        params_dict = {} if named_key is None else {named_key: {}}
        for k, v in vars(args).items():
            if not k.startswith(arg_prefix):
                continue
            k = k[len(arg_prefix):]
            items = k.split('.')
            _dict = params_dict
            for item in items[:-1]:
                _dict[item] = {}
                _dict = _dict[item]
            _dict[items[-1]] = v

        return self.merge_from_dict(params_dict, **kwargs)

    def merge_from_py(self, py_file_obj: object, named_key: str = None, **kwargs):
        """
        Args:
            py_file_obj (module): imported py_file_obj.
            named_key (str): named_key to wrap the parameters to be merged.
            override (bool): override the existed value if True, otherwise skip.
        """
        params_dict = {} if named_key is None else {named_key: {}}
        for k, v in vars(py_file_obj).items():
            if k.startswith('__'):
                continue
            params_dict[k] = v

        return self.merge_from_dict(params_dict, **kwargs)

    def merge_from_file(self, file_path: str, **kwargs):
        """
        Args:
            file_path (str): the path of config file.
            named_key (str): named_key to wrap the parameters to be merged.
            override (bool): override the existed value if True, otherwise skip.
        """
        if file_path.endswith(('.yaml', '.yml')):
            return self.merge_from_yaml(file_path, **kwargs)
        elif file_path.endswith('.json'):
            return self.merge_from_json(file_path, **kwargs)
        else:
            raise TypeError('{} ext format not support.'.format(file_path.split('.')[-1]))

    def merge_from_yaml(self, yaml_path: str, named_key: str = None, **kwargs):
        """
        Args:
            yaml_path (str): the path of yaml file.
            named_key (str): named_key to wrap the parameters to be merged.
            override (bool): override the existed value if True, otherwise skip.
        """
        def _load_yaml(_yaml_path):
            with open(_yaml_path, 'r') as f:
                return yaml.load(f, yaml.FullLoader)

        params = self if named_key is None else Parameters()
        self.__recursively_read_file(params, _load_yaml, yaml_path, **kwargs)
        if named_key is not None:
            setattr(self, named_key, params)

        return self

    def merge_from_json(self, json_path: str, named_key: str = None, **kwargs):
        """
        Args:
            json_path (str): the path of json file.
            named_key (str): named_key to wrap the parameters to be merged.
            override (bool): override the existed value if True, otherwise skip.
        """
        def _load_json(_json_path):
            with open(_json_path, 'r') as f:
                return json.load(f)

        params = self if named_key is None else Parameters()
        self.__recursively_read_file(params, _load_json, json_path, **kwargs)
        if named_key is not None:
            setattr(self, named_key, params)

        return self

    def merge_from_dict(self, params_dict: dict, override: bool = True):
        """
        Args:
            params_dict (dict): dictionary of parameters. Could be nested.
            override (bool): override the existed value if True, otherwise skip.
        """

        # def merge_a_dict(attr, dc, override):
        #     if isinstance(attr, Parameters):
        #         attr.merge_from_dict(v)
        #     elif attr is None or override:
        #         attr = Parameters().merge_from_dict(v)
        #
        #     return attr
        def merge_a_dict(src, qry, override_):
            # Check attr format.
            if not isinstance(src, (Parameters, dict, type(None))):
                if override_:
                    src = None
                else:
                    return src

            # Check output format
            qry = qry.copy()
            toParameters = qry.pop(Parameters.PARAMETERS_KEY, None)
            if toParameters is None or (not override_ and src is not None):
                toParameters = isinstance(src, Parameters)

            # Convert attr format & Merging
            if src is None:
                src = Parameters()
            elif isinstance(src, dict):
                src = Parameters().merge_from_dict(src)
            src.merge_from_dict(qry, override_)

            # Convert output format.
            if not toParameters:
                src = src.to_dict(False)

            return src

        assert isinstance(params_dict, dict), 'input must be a dict.'
        params_dict = params_dict.copy()

        if params_dict.pop(self.OVERWRITE_KEY, False):
            for k in list(vars(self)):
                delattr(self, k)
            override = True

        for key in params_dict.pop(Parameters.REMOVE_KEY, []):
            if hasattr(self, key):
                delattr(self, key)

        for key, value in params_dict.items():
            assert isinstance(key, str), 'key must be string.'
            if isinstance(value, dict):
                attr = getattr(self, key, None)
                attr = merge_a_dict(attr, value, override)
                setattr(self, key, attr)
            elif override or not hasattr(self, key):
                setattr(self, key, value)

        return self

    def to_dict(self, full=True):
        data = {}
        if full:
            data[Parameters.PARAMETERS_KEY] = True
        for k, v in self.items():
            if isinstance(v, Parameters):
                data[k] = v.to_dict(full)
            elif type(v).__name__ == 'module':
                continue
            else:
                data[k] = v

        return data

    def get(self, k, default=None):
        return getattr(self, k, default)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def copy(self):
        return copy.deepcopy(self)

    def dump(self, file_path: str):
        """
        Args:
            file_path (str): file_path where to dump parameters as yaml.
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def __recursively_read_file(params, file_reader, file_path, **kwargs):
        data = file_reader(file_path)
        assert isinstance(data, dict), 'file data must be a dict.'

        base_files = data.pop(Parameters.BASE_KEY, [])
        if isinstance(base_files, str):
            base_files = [base_files]
        for base_file in base_files:
            if not base_file.startswith('/'):
                base_file = os.path.abspath(os.path.join(os.path.dirname(file_path), base_file))
            assert base_file != os.path.abspath(file_path), '_BASE_ cannot point to self.'
            params.__recursively_read_file(params, file_reader, base_file, **kwargs)

        params.merge_from_dict(data, **kwargs)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(vars(self))

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        def recur_repr(param_dict, indent=0):
            sentence = ''

            for k, v in param_dict.items():
                if isinstance(v, dict) and v.pop(Parameters.PARAMETERS_KEY, False):
                    _sentence = recur_repr(v, indent+1)
                else:
                    _sentence = v.__repr__() + '\n'
                sentence += '{}{}: {}'.format('  '*indent, k, _sentence)

            return sentence

        return recur_repr(self.to_dict(True))


def is_number(string: str):
    try:
        float(string)
    except ValueError:
        return 0

    try:
        int(string)
    except ValueError:
        return -1

    return 1


class ArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)

        more_args = {}
        key = None
        for arg in argv:
            if arg.startswith('-'):
                key = arg.lstrip('-')
                more_args[key] = None
            elif key is not None:
                arg = self.auto_type(arg)
                if more_args[key] is None:
                    more_args[key] = arg
                elif isinstance(more_args[key], list):
                    more_args[key].append(arg)
                else:
                    more_args[key] = [more_args[key], arg]

        return args, more_args

    @staticmethod
    def auto_type(arg: str):
        try:
            return int(arg)
        except ValueError:
            pass

        try:
            return float(arg)
        except ValueError:
            pass

        return arg
