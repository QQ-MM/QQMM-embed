from typing import Optional
import importlib
from transformers import AutoTokenizer, AutoImageProcessor, ProcessorMixin

from qqmm.utils import load_config
from qqmm.utils.parameter_manage import Parameters


def build_processor(config: dict, inferring: bool = False) -> ProcessorMixin:
    """
    Build the processor for training.
    Args:
        config (dict):
            processor_class (str): Class of processor from "qqmm.models",
                e.g. "qqmm_ocular_qwen2.processing_qqmm.QQMMProcessor".
                Should be a subclass of transformers.ProcessorMixin
            tokenizer_path (str): Path to a huggingface pretrained model with a tokenizer.
            image_processor_path (str): Path to a huggingface pretrained model with an image processor.
            tokenizer_config (dict): Arguments to pass to transformers.AutoTokenizer.
            image_processor_config (dict): Arguments to pass to transformers.AutoImageProcessor.
            *: Additional arguments used to initialize the processor.
        inferring (bool): Inferring mode or not. When inferring mode, processor will not prepare dummy inputs.
    """
    config = config.copy()
    tokenizer = AutoTokenizer.from_pretrained(config.pop('tokenizer_path'),
                                              **config.pop('tokenizer_config', {}),
                                              trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(config.pop('image_processor_path'),
                                                         **config.pop('image_processor_config', {}),
                                                         trust_remote_code=True)

    processor_class = config.pop('processor_class', None)
    if processor_class is None:
        raise ValueError("Processor class (processor_class) is not given.")
    Processor = getattr(importlib.import_module(f"qqmm.models.{'.'.join(processor_class.split('.')[:-1])}"),
                        processor_class.split('.')[-1])

    processor = Processor(tokenizer=tokenizer, image_processor=image_processor, **config)
    processor.inferring = inferring

    return processor


def load_processor(name_or_path: str,
                   config: Optional[Parameters] = None,
                   inferring: bool = True) -> ProcessorMixin:
    """
    Load the processor from a checkpoint or a config file.
    Args:
        name_or_path (str): Path to a checkpoint or a config file.
        config (Parameters): QQMM config. If not provided, will try to use the config loaded from name_or_path.
        inferring (bool): Inferring mode or not.
    """
    if config is None:
        config = load_config(name_or_path)
    config.PROCESSOR_CONFIG.pop('image_token_len', None)
    processor = build_processor(config.PROCESSOR_CONFIG, inferring=inferring)

    return processor
