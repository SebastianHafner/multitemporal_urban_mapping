import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import numpy as np
import einops
from einops.layers.torch import Rearrange

from utils.experiment_manager import CfgNode


class Segmenter(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(Segmenter, self).__init__()

        # attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.patch_size = 8
        self.n_layers = 2
        self.n_heads = 3
        self.d_model = 192
        self.d_out = cfg.MODEL.OUT_CHANNELS

        # input and n patches
        assert (self.h % self.patch_size == 0)
        self.n_patches = self.h // self.patch_size

        # linear mapper
        self.patch_embedding = PatchEmbedding(self.c, self.patch_size, self.d_model)

        # positional embedding
        self.register_buffer('positional_encodings', get_positional_encodings(self.n_patches ** 2, self.d_model),
                             persistent=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=4 * self.d_model, batch_first=True,
                                                   activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)

        # decoding
        self.decoder = LinearDecoder(self.d_out, self.patch_size, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.size()
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')

        # tile each image and embed the resulting patches
        tokens = self.patch_embedding(x)

        # adding positional encoding
        out = tokens + self.positional_encodings.repeat(B, 1, 1)

        # transformer encoder
        out = self.encoder(out)

        out = self.decoder(out, (H, W))

        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=T)

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


def get_positional_encodings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class LinearDecoder(nn.Module):
    def __init__(self, n_cls, patch_size, hidden_d):
        super(LinearDecoder, self).__init__()

        self.hidden_d = hidden_d
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(hidden_d, n_cls)

    def forward(self, x, im_size: Tuple[int, int]) -> torch.Tensor:
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=GS)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


