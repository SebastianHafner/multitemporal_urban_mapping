import torch
import torch.nn as nn
import einops
from models import building_blocks as blocks
from utils.experiment_manager import CfgNode


class SimpleNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(SimpleNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS

        self.inc = blocks.InConv(n_channels, 64, blocks.DoubleConv)
        self.outc = blocks.OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        # x (TS, B, C, H, W)
        T, B, _, H, W = x.size()
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')
        out = self.outc(self.inc(x))
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=T)
        return out
