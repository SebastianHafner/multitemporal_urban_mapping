from typing import Tuple

import timm
import timm.models.vision_transformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from utils import experiment_manager


"""
Override timm.models.vision_transformer.VisionTransformer to
output all output tokens (excluding class or distill tokens)
This works for the vision_transformer_hybrid models as well
"""


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        """
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        """
        if self.dist_token is None:
            return x[:, 1:]
        else:
            return x[:, 2:]

    def forward(self, x):
        """
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
        """
        x = self.forward_features(x)
        return x


timm.models.vision_transformer.VisionTransformer = VisionTransformer


class MaskTransformer(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
        super(MaskTransformer, self).__init__()

        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        c = x[:, :self.num_classes]
        z = x[:, self.num_classes:]
        return z, c


class Upsample(nn.Module):
    def __init__(self, image_size: int, patch_size: Tuple[int, int]):
        super(Upsample, self).__init__()
        self.model = nn.Sequential(
            rearrange("b (p1 p2) c -> b c p1 p2", p1=image_size // patch_size[0], p2=image_size // patch_size[1]),
            nn.Upsample(scale_factor=patch_size, mode="bilinear")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Segmenter(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(Segmenter, self).__init__()

        num_classes = 2
        image_size = cfg.AUGMENTATION.CROP_SIZE

        self.cfg = cfg
        self.encoder = timm.create_model('vit_tiny_patch16_384', img_size=image_size, pretrained=True)
        patch_size = self.encoder.patch_embed.patch_size

        # input variables from table 1
        emb_dim = 768  # must be divisible by number of heads
        hidden_dim = 200
        num_layers = 12
        num_heads = 3

        self.mask_transformer = MaskTransformer(num_classes, emb_dim, hidden_dim, num_layers, num_heads)
        self.upsample = Upsample(image_size, patch_size)
        self.scale = emb_dim ** -0.5

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        z, c = self.mask_transformer(x)
        masks = z @ c.transpose(1, 2)
        masks = torch.softmax(masks / self.scale, dim=-1)
        return self.upsample(masks)


