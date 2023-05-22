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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, _, H, W = x.size()
        out_ch = []

        for t in range(T - 1):
            x_t1, x_t2 = x[:, t], x[:, t + 1]

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

            out_ch.append(out)

        out_ch = torch.stack(out_ch)
        out_ch = einops.rearrange(out_ch, 't b c h w -> b t c h w')
        return out_ch


class SiamDiffUNetPaper(nn.Module):
    """SiamUnet_diff segmentation network."""
    def __init__(self, cfg: CfgNode):
        super(SiamDiffUNetPaper, self).__init__()

        self.cfg = cfg
        self.input_nbr = cfg.MODEL.IN_CHANNELS
        self.label_nbr = cfg.MODEL.OUT_CHANNELS
        self.change_method = 'bitemporal'

        self.conv11 = nn.Conv2d(self.input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, self.label_nbr, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.size()
        out_ch = []

        for t in range(T - 1):
            x1, x2 = x[:, t], x[:, t + 1]
            """Forward method."""
            # Stage 1
            x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
            x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
            x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

            # Stage 2
            x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
            x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
            x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

            # Stage 3
            x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
            x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
            x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
            x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

            # Stage 4
            x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
            x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
            x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
            x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

            ####################################################
            # Stage 1
            x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
            x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
            x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

            # Stage 2
            x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
            x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
            x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

            # Stage 3
            x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
            x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
            x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
            x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

            # Stage 4
            x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
            x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
            x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
            x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

            # Stage 4d
            x4d = self.upconv4(x4p)
            pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
            x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
            x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
            x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
            x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

            # Stage 3d
            x3d = self.upconv3(x41d)
            pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
            x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
            x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
            x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
            x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

            # Stage 2d
            x2d = self.upconv2(x31d)
            pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
            x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
            x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
            x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

            # Stage 1d
            x1d = self.upconv1(x21d)
            pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
            x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
            x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
            x11d = self.conv11d(x12d)

            out_ch.append(x11d)

        out_ch = torch.stack(out_ch)
        out_ch = einops.rearrange(out_ch, 't b c h w -> b t c h w')

        return out_ch


class LUNetPaper(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(LUNetPaper, self).__init__()

        self.cfg = cfg
        self.img_ch = cfg.MODEL.IN_CHANNELS
        self.output_ch = cfg.MODEL.OUT_CHANNELS
        self.patch_size = cfg.AUGMENTATION.CROP_SIZE
        self.change_method = 'timeseries'

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=self.img_ch, ch_out=16)
        self.set1 = set_values(16, self.patch_size, self.patch_size)

        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.set2 = set_values(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.set3 = set_values(64, self.patch_size / 4, self.patch_size / 4)

        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.set4 = set_values(128, self.patch_size / 8, self.patch_size / 8)

        self.Conv5 = conv_block(ch_in=128, ch_out=256)
        self.set5 = set_values(256, self.patch_size / 16, self.patch_size / 16)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, self.output_ch, kernel_size=1, stride=1, padding=0)

    def encoder(self, x):
        x1, xout = self.set1(self.Conv1, x)

        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout)

        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout)

        x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout)

        x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout)

        return x1, x2, x3, x4, x5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, 'b t c h w -> t b c h w')
        # encoding path
        x1, x2, x3, x4, x5 = self.encoder(x)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


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

        # mapping (only for first and last image)
        features_seg = []
        for feature in features:
            feature = einops.rearrange(feature, '(b t) c h w -> t b c h w', t=T)
            first_last_feature_seg = feature[[0, -1]]
            first_last_feature_seg = einops.rearrange(first_last_feature_seg, 't b c h w -> (b t) c h w')
            features_seg.append(first_last_feature_seg)

        out_seg = self.outc_seg(self.decoder_seg(features_seg))
        out_seg = einops.rearrange(out_seg, '(b t) c h w -> t b c h w', t=2)
        out_seg1, out_seg2 = out_seg[0], out_seg[1]

        # change detection
        features_ch = []
        for feature, lstm_block in zip(features[::-1], self.lstm_blocks):
            feature_ch = lstm_block(einops.rearrange(feature, '(b t) c h w -> t b c h w', t=T))
            features_ch.append(feature_ch)

        out_ch = self.outc_ch(self.decoder_ch(features_ch[::-1]))

        return out_ch, out_seg1, out_seg2


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
        h_state, c_state = (
            Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).to(device),
            Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).to(device)
        )

        for t in range(xinp.size()[0]):
            input_t = xinp[t]
            h_state, c_state = self.RCell(input_t, h_state, c_state)

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




class MultiTaskLUNetPaper(nn.Module):
    def __init__(self, img_ch, output_ch, patch_size):
        super(MultiTaskLUNetPaper, self).__init__()

        self.patch_size = patch_size
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16)
        self.set1 = set_values(16, self.patch_size, self.patch_size)

        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.set2 = set_values(32, self.patch_size / 2, self.patch_size / 2)

        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.set3 = set_values(64, self.patch_size / 4, self.patch_size / 4)

        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.set4 = set_values(128, self.patch_size / 8, self.patch_size / 8)

        self.Conv5 = conv_block(ch_in=128, ch_out=256)
        self.set5 = set_values(256, self.patch_size / 16, self.patch_size / 16)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)
        self.Up5_segm = up_conv(ch_in=256, ch_out=128)
        self.Up_conv5_segm = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        self.Up4_segm = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4_segm = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.Up3_segm = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3_segm = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)
        self.Up2_segm = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2_segm = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_segm = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

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

        x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout)
        s4 = xout[0]
        a4 = xout[-1]

        x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout)
        s5 = xout[0]
        a5 = xout[-1]

        return x1, x2, x3, x4, x5, s1, s2, s3, s4, s5, a1, a2, a3, a4, a5

    def decoder_lstm(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def decoder_segm(self, s1, s2, s3, s4, s5):
        d5 = self.Up5_segm(s5)
        d5 = torch.cat((d5, s4), dim=1)
        d5 = self.Up_conv5_segm(d5)

        d4 = self.Up4_segm(d5)
        d4 = torch.cat((d4, s3), dim=1)
        d4 = self.Up_conv4_segm(d4)

        d3 = self.Up3_segm(d4)
        d3 = torch.cat((d3, s2), dim=1)
        d3 = self.Up_conv3_segm(d3)

        d2 = self.Up2_segm(d3)
        d2 = torch.cat((d2, s1), dim=1)
        d2 = self.Up_conv2_segm(d2)

        d1 = self.Conv_1x1_segm(d2)

        return d1

    def forward(self, input):
        x1, x2, x3, x4, x5, s1, s2, s3, s4, s5, a1, a2, a3, a4, a5 = self.encoder(input)

        d1 = self.decoder_lstm(x1, x2, x3, x4, x5)
        segm1 = self.decoder_segm(s1, s2, s3, s4, s5)
        segm2 = self.decoder_segm(a1, a2, a3, a4, a5)
        return d1, segm1, segm2


