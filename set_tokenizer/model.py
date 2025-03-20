# --------------------------------------------------------
# Model Structure of Set Tokenizer
#
# Based on TiTok (https://github.com/bytedance/1d-tokenizer/blob/main/modeling/titok.py)
# --------------------------------------------------------

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from set_tokenizer.module.blocks import Encoder, Decoder
from set_tokenizer.module.quantizer import VectorQuantizer
from set_tokenizer.module.maskgit_vqgan import Encoder as Pixel_Eecoder
from set_tokenizer.module.maskgit_vqgan import Decoder as Pixel_Decoder
from set_tokenizer.module.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer


@dataclass
class ModelArgs:
    codebook_size: int = 4096
    token_size: int = 12
    # vit arch
    vit_enc_model_size: str = "base"
    vit_dec_model_size: str = "base"
    vit_enc_patch_size: int = 16
    vit_dec_patch_size: int = 16
    num_latent_tokens: int = 64
    crop_size: int = 256


class tokenizer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.num_latent_tokens = config.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.quantize = VectorQuantizer(
            codebook_size=config.codebook_size,
            token_size=config.token_size)

        # Include MaskGiT-VQGAN's decoder
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256}))

    def encode(self, x):
        x = x * 0.5 + 0.5
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, tokenize_codes = self.quantize(z)
        return z_quantized, tokenize_codes

    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        decoded = self.pixel_decoder(decoded)
        decoded = 2.0 * decoded - 1.0
        return decoded

    def decode_code(self, tokens, shape=None, channel_first=True):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded


def SetTokenizer(**kwargs):
    return tokenizer(ModelArgs(**kwargs))