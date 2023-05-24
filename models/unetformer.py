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


class UNetFormer(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetFormer, self).__init__()

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
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV

        # transformers
        transformers = []
        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        for i, d_model in enumerate(transformer_dims):
            # positional encoding
            self.register_buffer(f'positional_encodings_{i}', get_positional_encodings(self.t, d_model),
                                 persistent=False)

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.n_heads,
                                                       dim_feedforward=self.d_hid, batch_first=True,
                                                       activation=self.activation)
            transformers.append(nn.TransformerEncoder(encoder_layer, self.n_layers))
        self.transformers = nn.ModuleList(transformers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.encoder(self.inc(x))

        # temporal modeling of features with transformer
        for i, transformer in enumerate(self.transformers):
            f = features[i]
            tokens = einops.rearrange(f, '(b1 b2) f h w -> (b1 h w) b2 f', b1=B)

            # adding positional encoding
            scale_factor = 2**(len(self.topology) - i)
            h, w = H // scale_factor, W // scale_factor
            tokens = tokens + getattr(self, f'positional_encodings_{i}').repeat(B * h * w, 1, 1)

            f = transformer(tokens)

            f = einops.rearrange(f, '(b1 h1 h2) t f -> (b1 t) f h1 h2', b1=B, h1=h)
            features[i] = f

        features = self.decoder(features)
        out = self.outc(features)
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        return out


class MultiTaskUNetFormer(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(MultiTaskUNetFormer, self).__init__()

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
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV

        self.decoder_ch = unet.Decoder(cfg)
        self.outc_ch = blocks.OutConv(self.topology[0], self.d_out)
        self.map_from_changes = cfg.MODEL.MAP_FROM_CHANGES
        self.adjacent_changes = cfg.MODEL.ADJACENT_CHANGES

        # transformers
        transformers = []
        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        for i, d_model in enumerate(transformer_dims):
            # positional encoding
            self.register_buffer(f'positional_encodings_{i}', get_positional_encodings(self.t, d_model),
                                 persistent=False)

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.n_heads,
                                                       dim_feedforward=self.d_hid, batch_first=True,
                                                       activation=self.activation)
            transformers.append(nn.TransformerEncoder(encoder_layer, self.n_layers))
        self.transformers = nn.ModuleList(transformers)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.encoder(self.inc(x))

        # temporal modeling of features with transformer
        for i, transformer in enumerate(self.transformers):
            f = features[i]
            tokens = einops.rearrange(f, '(b1 b2) f h w -> (b1 h w) b2 f', b1=B)

            # adding positional encoding
            scale_factor = 2 ** (len(self.topology) - i)
            h, w = H // scale_factor, W // scale_factor
            tokens = tokens + getattr(self, f'positional_encodings_{i}').repeat(B * h * w, 1, 1)

            f = transformer(tokens)

            f = einops.rearrange(f, '(b1 h1 h2) t f -> (b1 t) f h1 h2', b1=B, h1=h)
            features[i] = f

        # compute urban change detection features
        features_ch = []
        for feature in features:
            feature = einops.rearrange(feature, '(b1 b2) c h w -> b1 b2 c h w', b1=B)
            feature_ch = []
            for t in range(T - 1):
                if self.adjacent_changes:
                    feature_ch.append(feature[:, t + 1] - feature[:, t])
                else:
                    feature_ch.append(feature[:, -1] - feature[:, t])
            feature_ch = torch.stack(feature_ch)
            feature_ch = einops.rearrange(feature_ch, 't b c h w -> (b t) c h w')
            features_ch.append(feature_ch)

        # urban mapping
        features_sem = self.decoder(features)
        out_sem = self.outc(features_sem)
        out_sem = einops.rearrange(out_sem, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # urban change detection
        features_ch = self.decoder_ch(features_ch)
        out_ch = self.outc_ch(features_ch)
        out_ch = einops.rearrange(out_ch, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        if self.map_from_changes or self.training:
            return out_sem, out_ch
        else:
            return out_sem

    def continuous_mapping_from_logits(self, logits_sem: torch.Tensor, logits_ch: torch.Tensor,
                                       threshold: float = 0.5) -> torch.Tensor:
        assert self.map_from_changes
        y_hat_sem, y_hat_ch = torch.sigmoid(logits_sem) > threshold, torch.sigmoid(logits_ch) > threshold
        T = logits_sem.size(1)
        for t in range(T - 2, -1, -1):
            y_hat_sem[:, t] = y_hat_sem[:, t + 1] if self.adjacent_changes else y_hat_sem[:, -1]
            y_hat_sem[:, t][y_hat_ch[:, t]] = False
        return y_hat_sem.float()