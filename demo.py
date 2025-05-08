import sys
import os
import torch
from PIL import Image

from qqmm.models import build_processor
from qqmm.utils.parameter_manage import Parameters
from qqmm.models.qqmm_nav_qwen2.modeling_qqmm import QQMMForCausalLM
from qqmm.utils.chat import EmbedBot

config = Parameters()
config.merge_from_yaml('./configs/embed/qqmm-embed/mmeb.yaml')

print(">>> Building Model...")
processor = build_processor(config.PROCESSOR_CONFIG, inferring=True)
model = QQMMForCausalLM.from_pretrained('youzexue/QQMM-embed-v1', torch_dtype=torch.bfloat16, device_map='cuda')
bot = EmbedBot(model, processor)

print(">>> Inference...")
img = Image.open('assets/dog.png').convert("RGB")
img_feat = bot.chat(text='Represent the given image for classification.', image=[img])
txt_feat = bot.chat(text='Represent the following answer to an image classification task: a lovely dog.')
sim = (img_feat * txt_feat).sum()
print('Similarity score: ', sim)