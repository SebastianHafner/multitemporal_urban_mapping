import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import einops

from utils.experiment_manager import CfgNode

from models.embeddings import PatchEmbedding
from models.encodings import get_positional_encodings


class Segmenter(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(Segmenter, self).__init__()

        # attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.patch_size = cfg.MODEL.TRANSFORMER_PARAMS.PATCH_SIZE
        self.n_layers = cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
        self.n_heads = cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS
        self.d_model = cfg.MODEL.TRANSFORMER_PARAMS.D_MODEL
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.d_hid = self.d_model * 4
        self.activation = cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION

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
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)

        # decoding
        self.decoder = LinearDecoder(self.d_out, self.patch_size, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.size()
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')  # TODO: move this to patch embedding

        # tile each image and embed the resulting patches
        tokens = self.patch_embedding(x)

        # adding positional encoding
        out = tokens + self.positional_encodings.repeat(T * B, 1, 1)

        # transformer encoder
        out = self.encoder(out)

        out = self.decoder(out, (H, W))

        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=T)

        return out


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


