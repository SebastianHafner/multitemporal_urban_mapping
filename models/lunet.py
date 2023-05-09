# https://github.com/mpapadomanolaki/UNetLSTM
# https://github.com/mpapadomanolaki/multi-task-L-UNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import building_blocks as blocks
from utils import experiment_manager
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class set_values(nn.Module):
    def __init__(self, hidden_size, height, width, seq):
        super(set_values, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.RCell = blocks.RNNCell(self.hidden_size, self.hidden_size)
        self.seq = seq

    def forward(self, x: torch.tensor):
        T, BS = x.size()[:2]

        # collecting lstm outputs for each timestamp
        xout = Variable(torch.zeros(T, BS, self.hidden_size, self.height, self.width)).to(device)
        h_states = Variable(torch.zeros(T, BS, self.hidden_size, self.height, self.width)).to(device)

        h_state = Variable(torch.zeros(BS, self.hidden_size, self.height, self.width)).to(device)
        c_state = Variable(torch.zeros(BS, self.hidden_size, self.height, self.width)).to(device)

        for t in range(T):
            input_t = self.seq(x[t])
            xout[t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)
            h_states[t] = h_state

        return h_states, xout


class LUNetSmall(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(LUNetSmall, self).__init__()

        self.cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = blocks.DoubleConv(n_channels, 16)
        self.set1 = set_values(16, self.patch_size, self.patch_size, self.Conv1)

        self.Conv2 = blocks.DoubleConv(16, 32)
        self.set2 = set_values(32, self.patch_size / 2, self.patch_size / 2, nn.Sequential(self.Maxpool, self.Conv2))

        self.Conv3 = blocks.DoubleConv(32, 64)
        self.set3 = set_values(64, self.patch_size / 4, self.patch_size / 4, nn.Sequential(self.Maxpool, self.Conv3))

        self.Up3 = blocks.up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = blocks.DoubleConv(64, 32)

        self.Up2 = blocks.up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = blocks.DoubleConv(32, 16)

        self.Conv_1x1 = nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.tensor) -> tuple:
        T, BS, _, H, W = x.size()

        # encoder
        x1, xout = self.set1(x)
        x2, xout = self.set2(xout)
        x3, xout = self.set3(xout)

        # decoding all lstm outputs
        sout = Variable(torch.zeros(T, BS, 1, H, W)).to(device)
        for t in range(T):
            s_t = self.decoder(x1[t], x2[t], x3[t])
            sout[t] = s_t

        return sout

    def decoder(self, x1: torch.tensor, x2: torch.tensor, x3: torch.tensor) -> torch.tensor:

        s3 = self.Up3(x3)
        s3 = torch.cat((s3, x2), dim=1)
        s3 = self.Up_conv3(s3)

        s2 = self.Up2(s3)
        s2 = torch.cat((s2, x1), dim=1)
        s2 = self.Up_conv2(s2)

        s1 = self.Conv_1x1(s2)

        return s1


class LUNet(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(LUNet, self).__init__()

        self.cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE
        topology = [64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = blocks.DoubleConv(n_channels, topology[0])
        self.set1 = set_values(topology[0], self.patch_size, self.patch_size, self.Conv1)

        self.Conv2 = blocks.DoubleConv(topology[0], topology[1])
        self.set2 = set_values(topology[1], self.patch_size / 2, self.patch_size / 2, nn.Sequential(self.Maxpool, self.Conv2))

        self.Conv3 = blocks.DoubleConv(topology[1], topology[2])
        self.set3 = set_values(topology[2], self.patch_size / 4, self.patch_size / 4, nn.Sequential(self.Maxpool, self.Conv3))

        self.Conv4 = blocks.DoubleConv(topology[2], topology[3])
        self.set4 = set_values(topology[3], self.patch_size / 8, self.patch_size / 8, nn.Sequential(self.Maxpool, self.Conv4))

        self.Conv5 = blocks.DoubleConv(topology[3], 2 * topology[3])
        self.set5 = set_values(2 * topology[3], self.patch_size / 16, self.patch_size / 16, nn.Sequential(self.Maxpool, self.Conv5))

        self.Up5 = blocks.up_conv(ch_in=2 * topology[3], ch_out=topology[3])
        self.Up_conv5 = blocks.DoubleConv(2 * topology[3], topology[3])

        self.Up4 = blocks.up_conv(ch_in=topology[3], ch_out=topology[2])
        self.Up_conv4 = blocks.DoubleConv(topology[3], topology[2])

        self.Up3 = blocks.up_conv(ch_in=topology[2], ch_out=topology[1])
        self.Up_conv3 = blocks.DoubleConv(topology[2], topology[1])

        self.Up2 = blocks.up_conv(ch_in=topology[1], ch_out=topology[0])
        self.Up_conv2 = blocks.DoubleConv(topology[1], topology[0])

        self.Conv_1x1 = nn.Conv2d(topology[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, _, H, W = x.size()
        x = einops.rearrange(x, 'b t c h w -> t b c h w')

        # encoder
        x1, xout = self.set1(x)
        x2, xout = self.set2(xout)
        x3, xout = self.set3(xout)
        x4, xout = self.set4(xout)
        x5, xout = self.set5(xout)

        # decoding all lstm outputs
        sout = Variable(torch.zeros(T, B, 1, H, W)).to(device)
        for t in range(T):
            s_t = self.decoder(x1[t], x2[t], x3[t], x4[t], x5[t])
            sout[t] = s_t

        return sout

    def decoder(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor,
                x5: torch.Tensor) -> torch.Tensor:
        s5 = self.Up3(x5)
        s5 = torch.cat((s5, x4), dim=1)
        s5 = self.Up_conv3(s5)

        s4 = self.Up3(s5)
        s4 = torch.cat((s4, x3), dim=1)
        s4 = self.Up_conv3(s4)

        s3 = self.Up3(s4)
        s3 = torch.cat((s3, x2), dim=1)
        s3 = self.Up_conv3(s3)

        s2 = self.Up2(s3)
        s2 = torch.cat((s2, x1), dim=1)
        s2 = self.Up_conv2(s2)

        s1 = self.Conv_1x1(s2)

        return s1