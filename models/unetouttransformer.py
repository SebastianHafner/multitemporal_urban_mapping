import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import Tuple
import einops

from utils.experiment_manager import CfgNode

from models.embeddings import PatchEmbedding
from models import unet, encodings
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
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV

        # temporal encoding
        self.register_buffer('temporal_encodings', encodings.get_relative_encodings(self.t, self.d_model),
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

        # adding temporal encoding
        out = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        out = self.transformer(out)

        out = einops.rearrange(out, '(b1 h1 h2) t f -> b1 t f h1 h2', b1=B, h1=H)
        if self.training and self.disable_outc:
            return out

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
        self.outc_ch = blocks.OutConv(self.topology[0], self.d_out)
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV
        self.map_from_changes = cfg.MODEL.MAP_FROM_CHANGES

        # temporal encoding
        self.register_buffer('temporal_encodings', encodings.get_relative_encodings(self.t, self.d_model),
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

        # adding temporal encoding
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        features = self.transformer(tokens)

        features = einops.rearrange(features, '(b1 h1 h2) t f -> b1 t f h1 h2', b1=B, h1=H)

        # urban mapping
        out_sem = self.outc(einops.rearrange(features, 'b t f h w -> (b t) f h w'))
        out_sem = einops.rearrange(out_sem, '(b1 b2) c h w -> b1 b2 c h w', b1=B)

        # urban change detection
        out_ch = []
        for t in range(T - 1):
            out_ch.append(self.outc_ch(features[:, t + 1] - features[:, t]))
        out_ch.append(self.outc_ch(features[:, -1] - features[:, 0]))

        out_ch = einops.rearrange(torch.stack(out_ch), 't b c h w -> b t c h w')

        if self.map_from_changes or self.training:
            return out_sem, out_ch
        else:
            return out_sem

    def continuous_mapping_from_logits(self, logits_seg: torch.Tensor, logits_ch: torch.Tensor,
                                       threshold: float = 0.5) -> torch.Tensor:
        assert self.map_from_changes
        y_hat_seg, y_hat_ch = torch.sigmoid(logits_seg) > threshold, torch.sigmoid(logits_ch) > threshold
        T = logits_seg.size(1)
        for t in range(T - 2, -1, -1):
            y_hat_seg[:, t] = y_hat_seg[:, t + 1]  # use segmentation from previous timestamp
            y_hat_seg[:, t][y_hat_ch[:, t]] = False  # set changes to non-urban
        return y_hat_seg.float()


class UNetOutTransformerV4(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetOutTransformerV4, self).__init__()

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
        self.outc_seg = blocks.OutConv(self.topology[0], self.d_out)
        self.outc_ch = blocks.OutConv(self.topology[0], self.d_out)
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV
        self.map_from_changes = cfg.MODEL.MAP_FROM_CHANGES
        self.adjacent_changes = cfg.MODEL.ADJACENT_CHANGES

        # temporal encoding
        self.register_buffer('temporal_encodings', encodings.get_relative_encodings(self.t, self.d_model),
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
        out = einops.rearrange(out, '(b t) c h w -> b t c h w', b=B)

        # temporal modeling with transformer
        tokens = einops.rearrange(out, 'b t f h w -> (b h w) t f')

        # adding temporal encoding
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        features = self.transformer(tokens)

        features = einops.rearrange(features, '(b h w) t f -> b t f h w', b=B, h=H)

        return features


class UNetOutTransformerV5(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetOutTransformerV5, self).__init__()

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
        self.spatial_attention_size = cfg.MODEL.TRANSFORMER_PARAMS.SPATIAL_ATTENTION_SIZE

        # unet blocks
        self.inc = blocks.InConv(self.c, self.topology[0], blocks.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(self.topology[0], self.d_out)
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV

        # temporal encoding
        temporal_encodings = encodings.get_relative_encodings(self.t * self.spatial_attention_size**2, self.d_model)
        self.register_buffer('temporal_encodings', temporal_encodings, persistent=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()

        # feature extraction with unet
        # https://discuss.pytorch.org/t/how-to-extract-patches-from-an-image/79923/5

        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.decoder(self.encoder(self.inc(x)))
        out = einops.rearrange(out, '(b t) f h w -> b t f h w', b=B)

        # spatio-temporal modeling with transformer

        # patchify (b t c h w) -> (b t c h w hp hw)
        out = einops.rearrange(out, 'b t f h w -> (b t) f h w', b=B)
        out = transforms.Pad((1, 1, 1, 1), padding_mode='edge')(out)
        out = einops.rearrange(out, '(b t) f h w -> b t f h w', b=B)
        out = out.unfold(3, self.spatial_attention_size, 1).unfold(4, self.spatial_attention_size, 1)

        tokens = einops.rearrange(out, 'b t f h w ph pw-> (b h w) (t ph pw) f')

        # adding temporal encoding
        out = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # transformer encoder
        out = self.transformer(out)

        out = einops.rearrange(out, '(b h w) (t ph pw) f -> b t f h w ph pw', b=B, h=H, t=T,
                               ph=self.spatial_attention_size)
        out = out[:, :, :, :, :, self.spatial_attention_size // 2, self.spatial_attention_size // 2]
        if self.training and self.disable_outc:
            return out

        out = einops.rearrange(out, 'b t f h w -> (b t) f h w')
        out = self.outc(out)
        out = einops.rearrange(out, '(b t) c h w -> b t c h w', b=B)

        return out


class UNetOutTransformerV6(nn.Module):
    def __init__(self, cfg: CfgNode):
        # Super constructor
        super(UNetOutTransformerV6, self).__init__()

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
        self.spatial_attention_size = cfg.MODEL.TRANSFORMER_PARAMS.SPATIAL_ATTENTION_SIZE

        # unet blocks
        self.inc = blocks.InConv(self.c, self.topology[0], blocks.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(self.topology[0], self.d_out)
        self.disable_outc = cfg.MODEL.DISABLE_OUTCONV

        # positional encoding
        positional_encodings = encodings.get_relative_encodings(self.spatial_attention_size ** 2, self.d_model)
        self.register_buffer('positional_encodings', positional_encodings, persistent=False)

        temporal_encodings = encodings.get_relative_encodings(self.t, self.d_model)
        self.register_buffer('temporal_encodings', temporal_encodings, persistent=False)

        # transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_hid, batch_first=True,
                                                   activation=self.activation)
        # spatial encoder
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)
        # temporal encoder
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()

        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.decoder(self.encoder(self.inc(x)))
        out = einops.rearrange(out, '(b t) f h w -> b t f h w', b=B)

        # spatial attention
        # patchify (b t c h w) -> (b t c h w hp hw)
        out = einops.rearrange(out, 'b t f h w -> (b t) f h w', b=B)
        out = transforms.Pad((1, 1, 1, 1), padding_mode='edge')(out)
        out = einops.rearrange(out, '(b t) f h w -> b t f h w', b=B)
        out = out.unfold(3, self.spatial_attention_size, 1).unfold(4, self.spatial_attention_size, 1)

        tokens = einops.rearrange(out, 'b t f h w ph pw-> (b t h w) (ph pw) f')
        out = tokens + self.positional_encodings.repeat(B * T * H * W, 1, 1)
        out = self.spatial_transformer(out)

        out = einops.rearrange(out, '(b t h w) (ph pw) f -> b t f h w ph pw', b=B, t=T, h=H,
                               ph=self.spatial_attention_size)
        out = out[:, :, :, :, :, self.spatial_attention_size // 2, self.spatial_attention_size // 2]

        # temporal attention
        tokens = einops.rearrange(out, 'b t f h w -> (b h w) t f')
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)
        out = self.temporal_transformer(tokens)
        out = einops.rearrange(out, '(b h w) t f -> b t f h w', b=B, h=H)

        if self.training and self.disable_outc:
            return out

        out = einops.rearrange(out, 'b t f h w -> (b t) f h w')
        out = self.outc(out)
        out = einops.rearrange(out, '(b t) c h w -> b t c h w', b=B)

        return out
