from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from utils.segmenter_model.vit import VisionTransformer
from utils.segmenter_model.utils import checkpoint_filter_fn
from utils.segmenter_model.decoder import DecoderLinear, MaskTransformer
from utils.segmenter_model.segmenter import Segmenter
import utils.segmenter_model.torch as ptu
from utils.experiment_manager import CfgNode


def create_vit(cfg: CfgNode):
    backbone = 'vit_tiny_patch16_384'

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
    default_cfg["input_size"] = (cfg.MODEL.IN_CHANNELS, cfg.AUGMENTATION.CROP_SIZE, cfg.AUGMENTATION.CROP_SIZE)

    model = VisionTransformer(cfg)

    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, cfg: CfgNode):
    name = 'linear'
    if "linear" in name:
        decoder = DecoderLinear(n_cls=cfg.MODEL.OUT_CHANNELS, patch_size=encoder.patch_size, d_encoder=encoder.d_model)
    elif name == "mask_transformer":
        decoder_cfg = {'drop_path_rate': 0.0, 'dropout': 0.1, 'n_layers': 2}
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(n_cls=cfg.MODEL.OUT_CHANNELS, patch_size=encoder.patch_size,
                                  d_encoder=encoder.d_model, **decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(cfg: CfgNode):
    encoder = create_vit(cfg)
    decoder = create_decoder(encoder, cfg)
    model = Segmenter(encoder, decoder, n_cls=cfg.MODEL.OUT_CHANNELS)
    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant