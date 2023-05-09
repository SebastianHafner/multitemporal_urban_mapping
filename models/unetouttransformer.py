import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import einops

from utils.experiment_manager import CfgNode

from models.embeddings import PatchEmbedding
from models.encodings import get_positional_encodings
from models import unet
from models import building_blocks as blocks


class UNetOutTransformer(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetOutTransformer, self).__init__()

        # attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY
        self.n_layers = cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
        self.n_heads = cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS
        self.d_model = self.topology[0]
        self.d_hid = self.d_model * 4
        self.activation = cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION

        # unet blocks
        self.inc = blocks.InConv(self.c, self.topology[0], blocks.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(self.topology[0], self.d_out)

        # positional encoding
        self.register_buffer('positional_encodings', get_positional_encodings(self.t, self.d_model),
                             persistent=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.decoder(self.encoder(self.inc(x)))
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # temporal modeling with transformer
        tokens = einops.rearrange(out, 'b t f h w -> (b h w) t f')

        # adding positional encoding
        out = tokens + self.positional_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        out = self.transformer(out)

        out = einops.rearrange(out, '(b1 h1 h2) t f -> b1 t f h1 h2', b1=B, h1=H)

        out = einops.rearrange(out, 'b t f h w -> (b t) f h w')
        out = self.outc(out)
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        return out


class MultiTaskUNetOutTransformer(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(MultiTaskUNetOutTransformer, self).__init__()

        # attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY
        self.n_layers = cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
        self.n_heads = cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS
        self.d_model = self.topology[0]
        self.d_hid = self.d_model * 4
        self.activation = cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION

        # unet blocks
        self.inc = blocks.InConv(self.c, self.topology[0], blocks.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(self.topology[0], self.d_out)

        self.outc_ch = blocks.OutConv(self.topology[0] * 2, self.d_out)

        # positional encoding
        self.register_buffer('positional_encodings', get_positional_encodings(self.t, self.d_model),
                             persistent=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.decoder(self.encoder(self.inc(x)))
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # temporal modeling with transformer
        tokens = einops.rearrange(out, 'b t f h w -> (b h w) t f')

        # adding positional encoding
        tokens = tokens + self.positional_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        features = self.transformer(tokens)

        features = einops.rearrange(features, '(b1 h1 h2) t f -> b1 t f h1 h2', b1=B, h1=H)

        # urban mapping
        out_sem = self.outc(einops.rearrange(features, 'b t f h w -> (b t) f h w'))
        out_sem = einops.rearrange(out_sem, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # urban change detection
        out_ch = []
        for t in range(1, T):
            features_ch = torch.concat((features[:, t], features[:, t - 1]), dim=1)
            out_ch.append(self.outc_ch(features_ch))
        out_ch = einops.rearrange(torch.stack(out_ch), 't b c h w -> b t c h w')

        return out_sem, out_ch


class UNetOutTransformerSingleSeg(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetOutTransformerSingleSeg, self).__init__()

        # attributes
        self.cfg = cfg
        self.c = cfg.MODEL.IN_CHANNELS
        self.d_out = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY
        self.n_layers = cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
        self.n_heads = cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS
        self.d_model = self.topology[0]
        self.d_hid = self.d_model * 4
        self.activation = cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION

        # unet blocks
        self.inc = blocks.InConv(self.c, self.topology[0], blocks.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(self.topology[0], self.d_out)

        self.outc_ch = blocks.OutConv(self.topology[0] * 2, self.d_out)

        # positional encoding
        self.register_buffer('positional_encodings', get_positional_encodings(self.t, self.d_model),
                             persistent=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.decoder(self.encoder(self.inc(x)))
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # temporal modeling with transformer
        tokens = einops.rearrange(out, 'b t f h w -> (b h w) t f')

        # adding positional encoding
        tokens = tokens + self.positional_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        features = self.transformer(tokens)

        features = einops.rearrange(features, '(b1 h1 h2) t f -> b1 t f h1 h2', b1=B, h1=H)

        # urban mapping
        out_sem = self.outc(einops.rearrange(features, 'b t f h w -> (b t) f h w'))
        out_sem = einops.rearrange(out_sem, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # urban change detection
        out_ch = []
        for t in range(1, T):
            features_ch = torch.concat((features[:, t], features[:, t - 1]), dim=1)
            out_ch.append(self.outc_ch(features_ch))
        out_ch = einops.rearrange(torch.stack(out_ch), 't b c h w -> b t c h w')

        return out_sem, out_ch