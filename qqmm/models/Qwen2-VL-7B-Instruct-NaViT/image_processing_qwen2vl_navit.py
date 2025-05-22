# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/12/25 12:16:44
@email: fjjth98@163.com
@description:
Modified from transformers.models.qwen2_vl.image_processing_qwen2_vl.py, LICENSE:
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in this library. It has been modified from its
original forms to accommodate minor architectural differences compared
to GPT-NeoX and OPT used by the Meta AI team that trained the model.

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
from __future__ import annotations
from typing import Optional, Union

import numpy as np
from PIL import Image
from transformers.image_processing_utils import (BaseImageProcessor,
                                                 BatchFeature)
from transformers.image_transforms import (convert_to_rgb, resize,
                                           to_channel_dimension_format)
from transformers.image_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
                                      ChannelDimension, ImageInput,
                                      PILImageResampling,
                                      infer_channel_dimension_format,
                                      is_scaled_image, make_list_of_images,
                                      to_numpy_array, valid_images,
                                      validate_preprocess_arguments)
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


def find_best_size(
    image: Image.Image,
    size: dict[str, int],
    round_size: int,
    max_num_patches: int = -1
) -> tuple[int, int]:
    """Find best size for the given image

    Args:
        image (Image.Image): _description_
        size (dict[str, int]): {'shortest_edge': xxx, 'longest_edge': yyy}
        round_size (int): _description_
        max_num_patches (int, optional): _description_. Defaults to -1.

    Returns:
        tuple[int, int]: grid_size
    """
    # CAUTION: PIL.Image.Image.size return (width, height) rather than (height, width)
    width, height = image.size
    min_len, max_len = size['shortest_edge'], size['longest_edge']

    # min_len <= height, width <= max_len
    if width >= height and width > max_len:
        height, width = int(max_len * height / width), max_len
    elif height > width and height > max_len:
        height, width = max_len, int(max_len * width / height)
    height, width = max(height, min_len), max(width, min_len)

    # round (height, width) to k * round_size
    grid_height, grid_width = round(height / round_size), round(width / round_size)

    # make num_patches <= max_num_patches
    if 0 < max_num_patches < grid_height * grid_width:
        grid_height, grid_width = int((max_num_patches * height / width) ** 0.5), int((max_num_patches * width / height) ** 0.5)
        # ignore the aspect ratio if the resized grid_size is smaller than smallest one
        if grid_height * round_size < min_len:
            grid_height = min_len // round_size
            grid_width = max_num_patches // grid_height
        elif grid_width * round_size < min_len:
            grid_width = min_len // round_size
            grid_height = max_num_patches // grid_width

    return grid_height, grid_width


def make_list_of_images(
    images: list[Image.Image, list[Image.Image]],
    size: dict[str, int],
    patch_size: int,
    merge_size: int,
    max_single_num_patches: dict[str, int],
    max_total_num_patches: int = -1,
    min_resized_num_patches: int = -1
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    """_summary_

    Args:
        images (list[Image.Image, list[Image.Image]]): _description_
        size (dict[str, int]): _description_
        round_size (int): _description_
        max_single_num_patches (dict[str, int]): _description_
        max_total_num_patches (int): _description_
        min_resized_num_patches (int): _description_

    Returns:
        tuple[list[Image.Image], list[tuple[int, int]]]: flattened image list, grid_sizes
    """
    round_size = patch_size * merge_size
    merge_ratio = merge_size * merge_size
    max_single_image_num_grids = max_single_num_patches['image'] // merge_ratio
    max_single_video_num_grids = max_single_num_patches['video'] // merge_ratio
    max_total_num_grids = max_total_num_patches // merge_ratio
    min_resized_num_grids = min_resized_num_patches // merge_ratio

    min_grid_size = size['shortest_edge'] // round_size
    _images, grid_sizes = [], []
    # compute grid_sizes for all visual inputs
    for item in images:
        # image
        if isinstance(item, Image.Image):
            grid_height, grid_width = find_best_size(item, size, round_size, max_single_image_num_grids)
            _images.append(item)
            grid_sizes.append((grid_height, grid_width))
        # video: list[Image.Image]
        else:
            grid_height, grid_width = find_best_size(item[0], size, round_size, max_single_video_num_grids // len(item))
            _images += item
            grid_sizes += [(grid_height, grid_width)] * len(item)

    # reduce the grid_sizes if `total_num_grids` exceeds `max_total_num_grids`
    total_num_grids = sum(grid_height * grid_width for grid_height, grid_width in grid_sizes)
    if 0 < max_total_num_grids < total_num_grids:
        r2 = max_total_num_grids / total_num_grids
        for i, (grid_height, grid_width) in enumerate(grid_sizes):
            # if the resized image has less patches than the smallest one
            num_grids = grid_height * grid_width
            resized_num_grids = num_grids * r2
            if resized_num_grids <= min_resized_num_grids:
                if num_grids > min_resized_num_grids:
                    r = (min_resized_num_grids / num_grids) ** 0.5
                    resized_num_grids = min_resized_num_grids
                else:
                    continue
            else:
                r = r2 ** 0.5

            grid_height, grid_width = int(grid_height * r), int(grid_width * r)
            # ignore the aspect ratio if the resized grid_size is smaller than smallest one
            if grid_height < min_grid_size:
                grid_height, grid_width = min_grid_size, resized_num_grids // min_grid_size
            elif grid_width < min_grid_size:
                grid_height, grid_width = resized_num_grids // min_grid_size, min_grid_size
            grid_sizes[i] = (grid_height, grid_width)
    
    grid_sizes = [(grid_height * merge_size, grid_width * merge_size) for grid_height, grid_width in grid_sizes]

    return _images, grid_sizes


class Qwen2VLNaViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        merge_size: int = 2,
        max_single_num_patches: dict[str, int] = None,
        max_total_num_patches: int = None,
        min_resized_num_patches: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb
        self.max_single_num_patches = {'image': -1, 'video': -1} if max_single_num_patches is None else max_single_num_patches
        self.max_total_num_patches = -1 if max_total_num_patches is None else max_total_num_patches
        self.min_resized_num_patches = -1 if min_resized_num_patches is None else min_resized_num_patches

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        max_single_num_patches: dict[str, int] = None,
        max_total_num_patches: int = None,
        min_resized_num_patches: int = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            max_single_num_patches (`int`, *optional*, defaults to `None`):
                Max number of patches of one image / video
                {'image': 1024, 'video': -1 (-1 means no restriction)}
            max_total_num_patches (`int`, *optional*, defaults to `None`):
                Max number of patches of all images
            min_resized_num_patches (`int`, *optional*, defaults to `None`):
                Min number of patches of one image when max_total_num_patches is reached

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        max_single_num_patches = self.max_single_num_patches if max_single_num_patches is None else max_single_num_patches
        max_total_num_patches = self.max_total_num_patches if max_total_num_patches is None else max_total_num_patches
        min_resized_num_patches = self.min_resized_num_patches if min_resized_num_patches is None else min_resized_num_patches

        images, grid_sizes = make_list_of_images(
            images=images,
            size=size,
            patch_size=self.patch_size,
            merge_size=self.merge_size,
            max_single_num_patches=max_single_num_patches,
            max_total_num_patches=max_total_num_patches,
            min_resized_num_patches=min_resized_num_patches
        )

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                resize(image=image, size=(grid_height * self.patch_size, grid_width * self.patch_size), resample=resample, input_data_format=input_data_format)
                for image, (grid_height, grid_width) in zip(images, grid_sizes)
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format) for image in images
        ]

        # flatten image patches
        num_channels = images[0].shape[0]
        pixel_values = np.concatenate([
            image
            .reshape(num_channels, grid_height, self.patch_size, grid_width, self.patch_size)
            .transpose(1, 3, 0, 2, 4)
            .reshape(grid_height * grid_width, num_channels, self.patch_size, self.patch_size)
            for image, (grid_height, grid_width) in zip(images, grid_sizes)
        ], axis=0)  # (\sum_i h_i*w_i, C, P, P)

        data = {'pixel_values': pixel_values, 'grid_sizes': grid_sizes}

        return BatchFeature(data=data, tensor_type=return_tensors)

