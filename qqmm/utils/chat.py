import sys
import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(PROJECT_PATH)

from qqmm.utils.model_utils import to_device
from transformers.feature_extraction_utils import BatchFeature

from typing import Optional, Union, List, Dict, Tuple
from torch.utils.data.dataloader import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from io import BytesIO
import base64
from copy import deepcopy

VLConvo = List[Dict[str, Union[str, Image.Image]]]
ImageInput = Union[Image.Image, str, bytes]

def prepare_message(text='', image=[]):
    record = {}
    if image:
        instruction = '<image>\n' + text
        for _ in range(1, len(image)):
            instruction = '<image>\n' + instruction
        record['images'] = []
        for img in image:
            w, h = img.size
            if min(w, h) < 448:
                if w <= h:
                    img = img.resize((448, int(448/w*h)))
                else:
                    img = img.resize((int(448/h*w), 448))
            w, h = img.size
            if max(w, h) > 1512:
                if w >= h:
                    img = img.resize((1512, int(1512*h/w)))
                else:
                    img = img.resize((int(1512*w/h), 1512))
            record['images'].append(img)
    else:
        instruction = text                
    conversation = [
        {
            "from": "System",
            "value": "You are an AI assistant whose name is QQMM.\n\
- QQMM is a multi-modality conversational language model that is developed by Tencent QQ Team (腾讯QQ团队). It is designed to be helpful, honest, and harmless.\n\
- QQMM can understand and communicate fluently in the language chosen by the user such as English and 中文.\n\
- QQMM is capable of comprehending and articulating responses effectively based on the provided image."
        },
        {
            "from": "Human",
            "value": instruction
        }
    ]
    record['conversation'] = conversation
    return record

class EmbedBot:

    def __init__(self, model, processor):
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        model.generation_config.top_k=None
        self.model = model
        self.processor = processor
        self.past_key_values = None
        self.chat_template = "{% for message in messages %}\
{% if loop.first and messages[0]['role'] != 'system' %}\
{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}\
{% endif %}\
{% if (message['role'] != 'assistant') %}\
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}\
{% elif (message['role'] == 'assistant')%}\
{{'<|im_start|>' + message['role'] + '\n'}}\
{% generation %}{{message['content'] + '<|im_end|>' + '\n'}}{% endgeneration %}\
{% endif %}\
{% endfor %}\
{{ '<|im_start|>assistant\n' }}"

    @torch.inference_mode()
    def gen_embed(self,
                 input_ids,
                 *args,
                 max_new_tokens: int = 1,
                 do_sample: bool = False,
                 use_cache: bool = True,
                 return_dict_in_generate: bool = True,
                 output_hidden_states: bool = True,
                 **kwargs) -> torch.Tensor:
        input_ids = input_ids.to(self.model.device)
        args = to_device(args, device=self.model.device, dtype=self.model.dtype)
        kwargs = to_device(kwargs, device=self.model.device, dtype=self.model.dtype)
        
        # print(self.processor.batch_decode(input_ids)[0])
        outputs = self.model.generate(input_ids, *args, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                      use_cache=use_cache, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states, pad_token_id=self.processor.tokenizer.eos_token_id, **kwargs)
        embed = outputs['hidden_states'][0][-1][:, -1, :]
        embed_norm = torch.norm(embed, p=2, dim=-1)
        embed = embed / torch.clip(embed_norm, min=1e-7)
        embed = embed.float().to(torch.device('cpu'))
        return embed

    def chat(self,
             text: Optional[str] = '',
             image: Optional[Union[List[ImageInput], ImageInput]] = [],
             max_new_tokens: int = 1,
             do_sample: bool = False,
             use_cache: bool = True,
             return_dict_in_generate: bool = True,
             output_hidden_states: bool = True) -> torch.Tensor:
        
        qry_record = prepare_message(text=text, image=image if isinstance(image, list) else [image])
        qry_inputs = self.processor.apply_chat_template(qry_record['conversation'],
                                                            chat_template=self.chat_template,
                                                            tokenize=True,
                                                            add_generation_prompt=True,
                                                            return_assistant_tokens_mask=True,
                                                            return_dict=True)
        qry_inputs = self.processor(inputs=qry_inputs,
                                images=qry_record.get('images', None),
                                truncation=True,
                                max_length=4096)  
        _ = qry_inputs.pop('prompt_mask', None)
        embed = self.gen_embed(**qry_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                        use_cache=use_cache, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states)
        return embed