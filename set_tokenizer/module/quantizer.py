# --------------------------------------------------------
# Vector Quantizer Module
#
# Based on:
#   - Taming-Transformers (https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py)
#   - TiTok (https://github.com/bytedance/1d-tokenizer/blob/main/modeling/quantizer/quantizer.py)
# --------------------------------------------------------

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast


class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 ):
        super().__init__()
        self.temp = 1.0

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
        embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) 
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        z = torch.nn.functional.normalize(z, dim=-1)

        z_quantized = z + (z_quantized - z).detach()

        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        min_encoding_indices = min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])

        return z_quantized, min_encoding_indices

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
