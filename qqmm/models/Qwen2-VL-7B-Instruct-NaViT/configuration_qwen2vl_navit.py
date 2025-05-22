# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/12/25 12:14:20
@email: fjjth98@163.com
@description:
Modified from transformers.models.qwen2_vl.configuration_qwen2_vl.py, LICENSE:
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
================================================
"""
from transformers import PretrainedConfig


class Qwen2VLNaViTConfig(PretrainedConfig):
    model_type = "qwen2vl_navit"

    def __init__(
        self,
        depth=32,
        hidden_act="quick_gelu",
        hidden_size=1280,
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
