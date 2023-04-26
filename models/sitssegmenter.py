import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import Tuple
import einops

from utils.experiment_manager import CfgNode

from models.embeddings import TemporalPatchEmbedding
from models.encodings import get_positional_encodings
from models import building_blocks as blocks

class SITSSegmenter(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(SITSSegmenter, self).__init__()

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

        self.inc = blocks.InConv(self.c, 64, blocks.DoubleConv)
        self.outc = blocks.OutConv(64, self.d_out)

        # input and n patches
        assert (self.h % self.patch_size == 0)
        self.n_patches = cfg.DATALOADER.TIMESERIES_LENGTH

        # linear mapper
        self.patch_embedding = TemporalPatchEmbedding(self.c, self.patch_size, self.d_model)

        # positional embedding
        self.register_buffer('positional_encodings', get_positional_encodings(self.n_patches, self.d_model),
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

        # patches = einops.rearrange(x, 't b c (h s1) (w s2) -> (b h w) t (s1 s2 c)', s1=self.patch_size,
        #                            s2=self.patch_size)

        tokens = self.patch_embedding(x)

        # adding positional encoding
        out = tokens + self.positional_encodings.repeat(tokens.size(0), 1, 1)

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