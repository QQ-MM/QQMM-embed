from typing import Union, Optional, Tuple, List, Dict, Any

import PIL.Image
import torch

from .processing_utils import BaseVLProcessor


class QQMMProcessor(BaseVLProcessor):
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

        images = [image['image'] if isinstance(image, dict) else image for image in images]

        if len(images) > 0:
            pixel_values, grid_sizes = self.image_transform(images)
        else:
            pixel_values = torch.zeros((0, 3, self.image_processor.patch_size, self.image_processor.patch_size),
                                       dtype=torch.float32)
            grid_sizes = torch.zeros((0, 2), dtype=torch.int64)

        inputs['pixel_values'] = pixel_values
        inputs['grid_sizes'] = grid_sizes

        return inputs

    def image_transform(self, images: List[PIL.Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_inputs = self.image_processor(images, return_tensors='pt')
        pixel_values, grid_sizes = image_inputs['pixel_values'], image_inputs['grid_sizes']

        return pixel_values, grid_sizes

    def get_image_token_length(self, inputs: Dict[str, Any]) -> List[int]:
        grid_sizes = inputs.get('grid_sizes', None)
        if grid_sizes is None:
            return []
        grid_sizes = -(grid_sizes // -2)
        image_token_lens = (grid_sizes[:, 0] * grid_sizes[:, 1] + 1).tolist()

        return image_token_lens
