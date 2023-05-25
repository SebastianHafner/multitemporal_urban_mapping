import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.padding import ReplicationPad2d
from utils.experiment_manager import CfgNode
from models import unet
from models import building_blocks as blocks
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SiamDiffUNet(nn.Module):
    def __init__(self, cfg):
        super(SiamDiffUNet, self).__init__()
        self.cfg = cfg
        self.change_method = 'bitemporal'

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = blocks.InConv(n_channels, topology[0], blocks.DoubleConv)

        self.encoder = unet.Encoder(cfg)
        self.decoder = unet.Decoder(cfg)

        self.outc = blocks.OutConv(topology[0], n_classes)

    def _forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> torch.Tensor:
        x1_t1 = self.inc(x_t1)
        features_t1 = self.encoder(x1_t1)

        x1_t2 = self.inc(x_t2)
        features_t2 = self.encoder(x1_t2)

        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        x2 = self.decoder(features_diff)

        out = self.outc(x2)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, _, H, W = x.size()
        out_ch = []

        # change continuous
        for t in range(T - 1):
            out = self._forward(x[:, t], x[:, t + 1])
            out_ch.append(out)
        # change first-last
        out = self._forward(x[:, 0], x[:, -1])
        out_ch.append(out)

        out_ch = torch.stack(out_ch)
        out_ch = einops.rearrange(out_ch, 't b c h w -> b t c h w')

        return out_ch


class LUNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(LUNet, self).__init__()

        self.cfg = cfg
        self.img_ch = cfg.MODEL.IN_CHANNELS
        self.output_ch = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE
        self.change_method = 'timeseries'

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = blocks.InConv(n_channels, topology[0], blocks.DoubleConv)

        self.encoder = unet.Encoder(cfg)

        lstm_blocks = []
        for idx in range(len(topology) + 1):
            is_not_bottleneck = idx != len(topology)
            dim = topology[idx] if is_not_bottleneck else topology[-1]
            patch_size = self.patch_size // 2 ** idx
            lstm_blocks.append(FeatureLSTM(dim, patch_size, patch_size))
        self.lstm_blocks = nn.ModuleList(lstm_blocks)

        self.decoder = unet.Decoder(cfg)
        self.outc = blocks.OutConv(topology[0], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.inc(x)
        features = self.encoder(x)

        # change detection
        features_ch = []
        for feature, lstm_block in zip(features[::-1], self.lstm_blocks):
            feature_ch = lstm_block(einops.rearrange(feature, '(b t) c h w -> b t c h w', b=B))
            features_ch.append(feature_ch)

        out = self.decoder(features_ch[::-1])
        out = self.outc(out)

        return out


class MultiTaskLUNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(MultiTaskLUNet, self).__init__()

        self.cfg = cfg
        self.img_ch = cfg.MODEL.IN_CHANNELS
        self.output_ch = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE
        self.change_method = 'timeseries'

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = blocks.InConv(n_channels, topology[0], blocks.DoubleConv)

        self.encoder = unet.Encoder(cfg)

        lstm_blocks = []
        for idx in range(len(topology) + 1):
            is_not_bottleneck = idx != len(topology)
            dim = topology[idx] if is_not_bottleneck else topology[-1]
            patch_size = self.patch_size // 2**idx
            lstm_blocks.append(FeatureLSTM(dim, patch_size, patch_size))
        self.lstm_blocks = nn.ModuleList(lstm_blocks)

        self.decoder_ch = unet.Decoder(cfg)
        self.decoder_seg = unet.Decoder(cfg)

        self.outc_ch = blocks.OutConv(topology[0], n_classes)
        self.outc_seg = blocks.OutConv(topology[0], n_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _, H, W = x.size()
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.inc(x)
        features = self.encoder(x)

        # change detection
        features_ch = []
        for feature, lstm_block in zip(features[::-1], self.lstm_blocks):
            feature_ch = lstm_block(einops.rearrange(feature, '(b t) c h w -> b t c h w', b=B))
            features_ch.append(feature_ch)

        out_ch = self.decoder_ch(features_ch[::-1])
        out_ch = self.outc_ch(out_ch)

        out_seg = self.outc_seg(self.decoder_seg(features))
        out_seg = einops.rearrange(out_seg, '(b t) c h w -> b t c h w', b=B)

        return out_ch, out_seg


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):
        conc_inputs = torch.cat((input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state


class FeatureLSTM(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(FeatureLSTM, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.RCell = RNNCell(self.hidden_size, self.hidden_size)

    def forward(self, xinp: torch.Tensor):
        B, T, *_ = xinp.size()
        h_state, c_state = (
            Variable(torch.zeros(B, self.hidden_size, self.height, self.width)).to(device),
            Variable(torch.zeros(B, self.hidden_size, self.height, self.width)).to(device)
        )

        for t in range(T):
            h_state, c_state = self.RCell(xinp[:, t], h_state, c_state)

        return h_state


class set_values(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(set_values, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.dropout = nn.Dropout(0.7)
        self.RCell = RNNCell(self.hidden_size, self.hidden_size)

    def forward(self, seq, xinp):
        xout = Variable(
            torch.zeros(int(xinp.size()[0]), int(xinp.size()[1]), self.hidden_size, self.height, self.width)
        ).to(device)

        h_state, c_state = (
            Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).to(device),
            Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).to(device)
        )

        for t in range(xinp.size()[0]):
            input_t = seq(xinp[t])
            xout[t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)

        return self.dropout(h_state), xout


