import re
import torch
from torch import nn
from torch.nn import functional as F


def build_projection(projection_type: str, in_dim: int, out_dim: int) -> nn.Module:
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projection_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(in_dim, out_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_dim, out_dim))
        projection = nn.Sequential(*modules)
        return projection

    raise ValueError(f'Unknown projector type: {projection_type}')


class PerceiverProjection(nn.Module):
    def __init__(self, projection_type: str, in_dim: int, out_dim: int, exist_cls_embedding: bool = False):
        super().__init__()
        self.image_embedding = nn.Parameter(torch.empty(1, 4*in_dim))
        self.projection = build_projection(projection_type, 4*in_dim, out_dim)
        self.exist_cls_embedding = exist_cls_embedding

        nn.init.trunc_normal_(self.image_embedding, mean=0.0, std=0.02)

    def forward(self, input_embeds: torch.Tensor, grid_sizes: torch.Tensor):
        if input_embeds.size(0) == 0:
            embeds = torch.cat([self.image_embedding[:0], input_embeds.reshape(0, 4*input_embeds.size(-1))])
            return self.projection(embeds)

        grid_lens = grid_sizes[:, 0] * grid_sizes[:, 1]
        if self.exist_cls_embedding:
            grid_lens += 1

        d = input_embeds.size(-1)
        embeds_chunks = torch.split(input_embeds, grid_lens.tolist())
        embeds = []
        for i, x in enumerate(embeds_chunks):
            if self.exist_cls_embedding:
                x = x[1:]
            x = x.view(*grid_sizes[i], d)
            h, w = x.shape[:-1]
            if h % 2 or w % 2:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[:-1]
            x = x.view(h//2, 2, w//2, 2, d).permute(0, 2, 1, 3, 4).reshape(h//2 * w//2, 4*d)
            embeds += [self.image_embedding, x]
        embeds = torch.cat(embeds)

        embeds = self.projection(embeds)

        return embeds
