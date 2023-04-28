import torch
import torch.nn as nn
from torch.autograd import Variable
import einops
from collections import OrderedDict
from models import building_blocks as blocks
from utils import experiment_manager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(UNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = blocks.InConv(n_channels, topology[0], blocks.DoubleConv)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.outc = blocks.OutConv(topology[0], n_classes)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        B, T, _, H, W = x.size()
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        out = self.outc(self.decoder(self.encoder(self.inc(x))))
        out = einops.rearrange(out, '(b1 b2) c h w -> b1 b2 c h w', b1=B)
        return out


class Encoder(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(Encoder, self).__init__()

        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = blocks.Down(in_dim, out_dim, blocks.DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


class Decoder(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(Decoder, self).__init__()
        self.cfg = cfg

        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = blocks.Up(in_dim, out_dim, blocks.DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1


