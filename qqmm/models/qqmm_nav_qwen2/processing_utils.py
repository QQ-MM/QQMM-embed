from typing import Union, Optional, List, Dict, Any
import warnings
from collections import UserDict, OrderedDict
import copy

import PIL.Image
import torch
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import TextInput
from transformers.image_utils import ImageInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType


class BaseProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    model_input_names = ["input_ids", "attention_mask"]
    tokenizer_class = "AutoTokenizer"
    ROLE_MAP = {'Human': 'user', 'AI': 'assistant', 'System': 'system'}

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)

        if self.chat_template is None:
            self.chat_template = self.tokenizer.chat_template
        self.inferring = True
        self.quiet = False
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], *args, **kwargs):
        conversation = copy.deepcopy(conversation)
        for msg in conversation:
            if 'from' in msg:
                msg['role'] = self.ROLE_MAP[msg.pop('from')]
            if 'value' in msg:
                msg['content'] = msg.pop('value')

        return super().apply_chat_template(conversation, *args, **kwargs)

    def __call__(self, *args, inputs: Optional[Union[Dict[str, Any], UserDict]] = None, **kwargs) -> BatchFeature:
        if inputs is None:
            inputs = {}
        elif isinstance(inputs, UserDict):
            inputs = inputs.data

        inputs = self.process(*args, inputs=inputs, **kwargs)

        return BatchFeature(inputs)

    def process(
        self,
        text: Union[TextInput] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
        return_tensors: TensorType = TensorType.PYTORCH,
        **kwargs
    ) -> Dict[str, Any]:
        assert return_tensors is TensorType.PYTORCH, "return_tensors must be TensorType.PYTORCH or 'pt'."

        if inputs is None:
            inputs = {}
        elif isinstance(inputs, UserDict):
            inputs = inputs.data

        if 'input_ids' not in inputs:
            input_ids = self.tokenizer(text, padding=False, truncation=False, return_attention_mask=False,
                                       **kwargs)['input_ids']
            inputs['input_ids'] = input_ids

        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = [1] * len(inputs['input_ids'])

        if 'assistant_masks' in inputs:
            inputs['prompt_mask'] = [1-x for x in inputs.pop('assistant_masks')]

        inputs = self.process_inputs(inputs)

        if truncation and len(inputs['input_ids']) > max_length:
            inputs = self.truncate(inputs, max_length)

        if padding and len(inputs['input_ids']) < max_length:
            inputs = self.padding(inputs, max_length)

        inputs = self.to_tensor(inputs)

        return inputs

    def process_inputs(self, inputs: Dict[str, Any]):
        return inputs

    def truncate(self, inputs: Dict[str, Any], max_length: int):
        if not self.quiet:
            warnings.warn(f"Truncate too long text.")
        inputs['input_ids'] = inputs['input_ids'][:max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:max_length]
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = inputs['prompt_mask'][:max_length]

        return inputs

    def padding(self, inputs: Dict[str, Any], max_length: int):
        padding_len = max_length - len(inputs['input_ids'])
        inputs['input_ids'] += [self.pad_token_id] * padding_len
        inputs['attention_mask'] += [0] * padding_len
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] += [0] * padding_len

        return inputs

    def to_tensor(self, inputs):
        inputs['input_ids'] = torch.tensor([inputs['input_ids']], dtype=torch.long)
        inputs['attention_mask'] = torch.tensor([inputs['attention_mask']], dtype=torch.bool)
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = torch.tensor([inputs['prompt_mask']], dtype=torch.bool)

        return inputs

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs):
        text = self.tokenizer.decode(token_ids, **kwargs)

        return text

    def batch_decode(self, sequences: Union[List[List[int]], torch.Tensor], **kwargs):
        text = self.tokenizer.batch_decode(sequences, **kwargs)

        return text

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def special_tokens(self):
        return [token.content for token in self.tokenizer.added_tokens_decoder.values()]


class BaseVLProcessor(BaseProcessor):
    attributes = BaseProcessor.attributes + ["image_processor"]
    optional_attributes = BaseProcessor.optional_attributes + ["image_token_len", "vision_token_share_pe"]
    model_input_names = BaseProcessor.model_input_names + ["pixel_values"]
    image_processor_class = "AutoImageProcessor"

    image_token = '<image>'

    def __init__(self, tokenizer: PreTrainedTokenizer, image_processor: BaseImageProcessor,
                 image_token_len: int = None, vision_token_share_pe: bool = True, **kwargs):
        super().__init__(tokenizer=tokenizer, image_processor=image_processor,
                         image_token_len=image_token_len, vision_token_share_pe=vision_token_share_pe,
                         **kwargs)

        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.image_token]},
                                          replace_additional_special_tokens=False)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        size_d = self.image_processor.size
        if 'shortest_edge' in size_d:
            image_size = (size_d['shortest_edge'], size_d['shortest_edge'])
        else:
            image_size = (size_d['height'], size_d['width'])
        assert image_size[0] == image_size[1]
        self.image_size = image_size

    def process(
        self,
        text: Union[TextInput] = None,
        images: Optional[ImageInput] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
        return_tensors: TensorType = TensorType.PYTORCH,
        **kwargs
    ) -> Dict[str, Any]:

        if inputs is None:
            inputs = {}
        elif isinstance(inputs, UserDict):
            inputs = inputs.data

        inputs = self.process_images(images, inputs=inputs)

        inputs = super().process(text=text, padding=padding, truncation=truncation, max_length=max_length,
                                 inputs=inputs, return_tensors=return_tensors, **kwargs)

        self.check(inputs)

        if self.vision_token_share_pe:
            position_ids = self.get_position_ids(inputs)
            position_ids = torch.tensor([position_ids], dtype=torch.long)
            inputs['position_ids'] = position_ids

        return inputs

    def process_images(self,
                       images: Optional[Union[List[Dict], List[PIL.Image.Image], PIL.Image.Image]],
                       inputs: Dict[str, Any]):
        assert isinstance(inputs, dict), \
            f"For process_image in OcularProcessor, inputs must be given as a dict rather than {type(inputs)}"

        if images is None:
            images = []
        elif isinstance(images, PIL.Image.Image):
            images = [images]

        if len(images) == 0 and self.inferring:
            return inputs

        pixel_values = []
        for image in images:
            if isinstance(image, dict):
                image = image['image']

            image = self.image_transform(image)

            pixel_values.append(image)

        if len(pixel_values) > 0:
            pixel_values = torch.stack(pixel_values, dim=0)
        else:
            pixel_values = torch.zeros((0, 3) + self.image_size, dtype=torch.float32)

        inputs['pixel_values'] = pixel_values

        return inputs

    def image_transform(self, image: PIL.Image.Image) -> torch.Tensor:
        image = self.image_processor(image, return_tensors='pt')

        return image

    def process_inputs(self, inputs: Dict[str, Any]):
        graft_token_lens = self._get_graft_token_length(inputs)

        inputs['input_ids'] = self._graft_token(inputs['input_ids'], graft_token_lens, self.image_token_id)
        inputs['attention_mask'] = self._graft_token(inputs['attention_mask'], graft_token_lens, 'replicate')
        if 'prompt_mask' in inputs:
            inputs['prompt_mask'] = self._graft_token(inputs['prompt_mask'], graft_token_lens, 'replicate')

        return inputs

    def truncate(self, inputs: Dict[str, Any], max_length: int):
        assert self.image_token_id not in inputs['input_ids'][max_length:], f"Truncate image token is not allowed."

        return super().truncate(inputs, max_length)

    def check(self, inputs: Dict[str, Any]):
        image_embed_token_count = torch.count_nonzero(inputs['input_ids'] == self.image_token_id).item()
        image_embed_count = sum(self.get_image_token_length(inputs))
        assert image_embed_token_count == image_embed_count, \
            "Wrong image embed token count, " \
            f"image_embed_token_count({image_embed_token_count}) != image_embed_count({image_embed_count})"

    def get_image_token_length(self, inputs: Dict[str, Any]) -> List[int]:
        num_images = len(inputs['pixel_values']) if inputs['pixel_values'] is not None else 0
        return [self.image_token_len] * num_images

    def _get_graft_token_length(self, inputs: Dict[str, Any]) -> Dict[int, int]:
        image_token_pos = [i for i, x in enumerate(inputs['input_ids']) if x == self.image_token_id]
        image_token_lens = self.get_image_token_length(inputs)

        assert len(image_token_pos) == len(image_token_lens), \
            "Wrong image token count, " \
            f"image_token_count({len(image_token_pos)}) != image_count({len(image_token_lens)})"

        graft_token_lens = OrderedDict(item for item in zip(image_token_pos, image_token_lens))

        return graft_token_lens

    @staticmethod
    def _graft_token(seq: List[int], graft_token_lens: Dict[int, int], value: Union[int, str]):
        if value == 'replicate':
            for i in reversed(graft_token_lens.keys()):
                seq[i:] = [seq[i]] * graft_token_lens[i] + seq[i+1:]
        else:
            for i in reversed(graft_token_lens.keys()):
                seq[i:] = [value] * graft_token_lens[i] + seq[i+1:]

        return seq

    def get_position_ids(self, inputs: Dict[str, Any]):
        input_ids = inputs['input_ids'][0]
        image_token_lens = self.get_image_token_length(inputs)
        position_ids = []
        i, j = 0, 0
        while len(position_ids) < len(input_ids):
            if input_ids[len(position_ids)] == self.image_token_id:
                position_ids += [i] * image_token_lens[j]
                j += 1
            else:
                position_ids.append(i)
            i += 1

        assert j == len(image_token_lens) and len(position_ids) == len(input_ids), \
            f"Wrong position_ids, {j} != {len(image_token_lens)} or {len(position_ids)} != {len(input_ids)}"

        return position_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = self._recover_image_token_id(token_ids)
        text = super().decode(token_ids, **kwargs)

        return text

    def batch_decode(self, sequences: Union[List[List[int]], torch.Tensor], **kwargs):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        sequences = [self._recover_image_token_id(token_ids) for token_ids in sequences]
        texts = super().batch_decode(sequences, **kwargs)

        return texts

    def _recover_image_token_id(self, token_ids):
        i = 0
        while True:
            if token_ids[i] == self.image_token_id:
                j = i + 1
                while j < len(token_ids) and token_ids[j] == self.image_token_id:
                    token_ids.pop(j)
            i += 1
            if i >= len(token_ids):
                break

        return token_ids

    @property
    def special_tokens(self):
        special_tokens = super().special_tokens
        special_tokens.remove(self.image_token)
        return special_tokens

