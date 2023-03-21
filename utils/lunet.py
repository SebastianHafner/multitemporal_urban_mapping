# https://github.com/mpapadomanolaki/UNetLSTM
# https://github.com/mpapadomanolaki/multi-task-L-UNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import network_building_blocks as blocks
from utils import experiment_manager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class set_values(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(set_values, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.RCell = blocks.RNNCell(self.hidden_size, self.hidden_size)

    def forward(self, seq, xinp):
        xout = Variable(torch.zeros(xinp.size()[0], xinp.size()[1], self.hidden_size, self.height, self.width))\
            .to(device)
        h_state = Variable(torch.zeros(xinp[0].shape[0], self.hidden_size, self.height, self.width)).to(device)
        c_state = Variable(torch.zeros(xinp[0].shape[0], self.hidden_size, self.height, self.width)).to(device)

        for t in range(xinp.size()[0]):
            input_t = seq(xinp[t])
            xout[t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)

        return h_state, xout


class LUNet(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode):
        super(LUNet, self).__init__()

        self.cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = blocks.DoubleConv(n_channels, 16)
        self.set1 = set_values(16, self.patch_size, self.patch_size)

        self.Conv2 = blocks.DoubleConv(16, 32)
        self.set2 = set_values(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = blocks.DoubleConv(32, 64)
        self.set3 = set_values(64, self.patch_size / 4, self.patch_size / 4)

        self.Up3 = blocks.up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = blocks.DoubleConv(64, 32)
        self.Up3_segm = blocks.up_conv(ch_in=64, ch_out=32)
        self.Up_conv3_segm = blocks.DoubleConv(64, 32)

        self.Up2 = blocks.up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = blocks.DoubleConv(32, 16)
        self.Up2_segm = blocks.up_conv(ch_in=32, ch_out=16)
        self.Up_conv2_segm = blocks.DoubleConv(32, 16)

        self.Conv_1x1 = nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_segm = nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, input: torch.tensor) -> tuple:
        # input (TS, BS, C, H, W)
        x1, x2, x3, s1, s2, s3, a1, a2, a3 = self.encoder(input)

        d1 = self.decoder_lstm(x1, x2, x3)
        segm1 = self.decoder_segm(s1, s2, s3)
        segm2 = self.decoder_segm(a1, a2, a3)
        return d1, segm1, segm2

    def encoder(self, x):
        x1, xout = self.set1(self.Conv1, x)
        s1 = xout[0]
        a1 = xout[-1]

        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout)
        s2 = xout[0]
        a2 = xout[-1]

        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout)
        s3 = xout[0]
        a3 = xout[-1]

        return x1, x2, x3, s1, s2, s3, a1, a2, a3,

    def decoder_lstm(self, x1, x2, x3):
        d3 = self.Up3(x3)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def decoder_segm(self, s1, s2, s3):
        d3 = self.Up3_segm(s3)
        d3 = torch.cat((d3, s2), dim=1)
        d3 = self.Up_conv3_segm(d3)

        d2 = self.Up2_segm(d3)
        d2 = torch.cat((d2, s1), dim=1)
        d2 = self.Up_conv2_segm(d2)

        d1 = self.Conv_1x1_segm(d2)

        return d1