import torch
import torch.nn as nn
from torchvision import models
import einops
from models import building_blocks as blocks
from utils.experiment_manager import CfgNode


class ResNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(ResNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS

        model = models.resnet152(pretrained=True)
        self.model = torch.nn.Sequential(*(list(model.children())[:-3]))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        # x (TS, B, C, H, W)
        T, B, _, H, W = x.size()
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')
        out = self.model(x)
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=T)
        return out
