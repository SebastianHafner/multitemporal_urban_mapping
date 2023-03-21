import torch
import torch.nn as nn
from utils import experiment_manager


class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, cfg: experiment_manager.CfgNode, encoder_type: str = 'ESPNet_C', classes: int = 19, p: int = 2,
                 q: int = 3, encoder_file=None):
        '''
        :param classes: number of classes in the dataset. Default is 19 for the cityscapes_video
        # does not work for fewer than 5 classes -> out conv converts to 6
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        self.cfg = cfg
        if encoder_type == 'ESPNet_C_L1b':
            self.encoder = ESPNet_C_L1b(cfg, p, q)
        elif encoder_type == 'ESPNet_C':
            self.encoder = ESPNet_Encoder(classes, p, q)
        else:
            assert False
        if encoder_file != None:
            self.encoder.load_state_dict(torch.load(encoder_file))
            print('Encoder loaded!')
        # load the encoder modules
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(16 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)
        )

        self.combine_l2_l3 = nn.Sequential(
            BR(2 * classes),
            DilatedParllelResidualBlockB(2 * classes, classes, add=False)
        )

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False),
            BR(classes)
        )

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)

        self.outc = nn.Conv2d(classes, 1, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        # [TS, BS, C, H, W]
        # input = input.squeeze(0)  # 5Ax -> 4Ax

        # Conv-3_red
        output0 = self.modules[0](input)
        # RGB_1, down-scaled by recursive avg-pooling
        inp1 = self.modules[1](input)
        # RGB_2, down-scaled by recursive 0avg-pooling
        inp2 = self.modules[2](input)

        # Concat_0
        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))

        # down-sampled, ESP_red_0
        output1_0 = self.modules[4](output0_cat)

        # p times ESP_0
        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        # Concat_1
        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))

        # down-sampled, ESP_red_1
        output2_0 = self.modules[7](output1_cat)

        # q times ESP_1
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        # concatenate for feature map width expansion, Concat_2
        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))

        # RUM, Conv-1_2 + DeConv_green_0
        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))

        # project to C-dimensional space, Conv-1_1
        output1_C = self.level3_C(output1_cat)

        # RUM, Concat_3 + ESP_2 + DeConv_green_1
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1)))

        # Concat_4 + Conv-1
        concat_features = self.conv(torch.cat([comb_l2_l3, output0], 1))

        # DeConv_green_2
        classifier = self.classifier(concat_features)

        # classifier = F.softmax(classifier, dim=1)
        # classifier = classifier.unsqueeze(0)  # 4Ax -> 5Ax

        out = self.outc(classifier)
        return out

    def is_lstm_net(self) -> bool:
        return False


class ESPNet_L1b(ESPNet):

    def __init__(self, cfg: experiment_manager.CfgNode, encoder_file=None):
        super().__init__(cfg, encoder_type='ESPNet_C_L1b', encoder_file=encoder_file)

    def forward(self, input: torch.tensor, states: torch.tensor) -> torch.tensor:
        # Dimensions: Time, BatchSize, Channels, Height, Width
        input = input.view(-1, input.shape[2], input.shape[3], input.shape[4])  # merge time and bs dim

        output0 = self.modules[0](input)  # Conv-3_red
        inp1 = self.modules[1](input)  # RGB_1, down-scaled by recursive avg-pooling
        inp2 = self.modules[2](input)  # RGB_2, down-scaled by recursive 0avg-pooling
        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))  # Concat_0
        output1_0 = self.modules[4](output0_cat)  # down-sampled, ESP_red_0

        for i, layer in enumerate(self.modules[5]):  # p times ESP_0
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))  # Concat_1

        output2_0 = self.modules[7](output1_cat)  # down-sampled, ESP_red_1
        for i, layer in enumerate(self.modules[8]):  # q times ESP_1
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[9](
            torch.cat([output2_0, output2], 1))  # concatenate for feature map width expansion, Concat_2

        if self.encoder.is_batch_norm:
            batch_norm_features = self.modules[10](output2_cat)
            lstm_in = batch_norm_features.view(-1, self.cfg.TRAINER.BATCH_SIZE, batch_norm_features.shape[1],
                                               batch_norm_features.shape[2],
                                               batch_norm_features.shape[3])  # -1 ... time_steps, 1..bs
            lstm_out, new_states = self.modules[11](lstm_in, states)
            lstm_out = lstm_out.view(-1, lstm_out.shape[2], lstm_out.shape[3], lstm_out.shape[4])
        else:
            batch_norm_features = output2_cat
            lstm_in = batch_norm_features.view(-1, self.cfg.TRAINER.BATCH_SIZE, batch_norm_features.shape[1],
                                               batch_norm_features.shape[2],
                                               batch_norm_features.shape[3])  # -1 ... time_steps, 1..bs
            lstm_out, new_states = self.modules[10](lstm_in, states)
            lstm_out = lstm_out.view(-1, lstm_out.shape[2], lstm_out.shape[3], lstm_out.shape[4])

        output2_c = self.up_l3(self.br(lstm_out))  # RUM, Conv-1_2 + DeConv_green_0

        output1_C = self.level3_C(output1_cat)  # project to C-dimensional space, Conv-1_1
        comb_l2_l3 = self.up_l2(
            self.combine_l2_l3(torch.cat([output1_C, output2_c], 1)))  # RUM, Concat_3 + ESP_2 + DeConv_green_1

        concat_features = self.conv(torch.cat([comb_l2_l3, output0], 1))  # Concat_4 + Conv-1

        classifier = self.classifier(concat_features)  # DeConv_green_2
        classifier = classifier.view(-1, self.cfg.TRAINER.BATCH_SIZE, classifier.shape[1], classifier.shape[2],
                                     classifier.shape[3])  # -1 ... time_steps, 1..bs
        out = self.outc(classifier.squeeze(0))
        return out.unsqueeze(0), new_states

    def update_parameters(self, batch_size, time_steps, overlap):
        self.encoder.update_parameters(batch_size, time_steps, overlap)

    def is_lstm_net(self) -> bool:
        return True


class ESPNet_C_L1b(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode, p: int = 2, q: int = 3):
        super().__init__()
        self.state_scale_factor = 8
        self.state_channels = 16

        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)

        # LSTM stuff
        lstm_activation_function = 'prelu'
        self.clstm = ConvLSTM(256, 19, kernel_size=5, activation_function=lstm_activation_function, state_init='learn',
                              cell_type=5, batch_size=cfg.TRAINER.BATCH_SIZE,
                              time_steps=cfg.DATALOADER.TIMESERIES_LENGTH,
                              state_img_size=[cfg.AUGMENTATION.CROP_SIZE // 8, cfg.AUGMENTATION.CROP_SIZE // 8])
        self.is_batch_norm = False
        if lstm_activation_function == 'tanh':
            self.is_batch_norm = True
        if self.is_batch_norm:
            self.batch_norm = nn.BatchNorm2d(256)

    def forward(self, input, states):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        input = input.contiguous().view(-1, input.shape[2], input.shape[3], input.shape[4])  # merge time and bs dim

        output0 = self.level1(input)  # 512x1024 --> 256x512
        inp1 = self.sample1(input)  # scale down RGB
        inp2 = self.sample2(input)  # scale down RGB

        output0_cat = self.b1(torch.cat([output0, inp1], 1))  # Concat_0
        output1_0 = self.level2_0(output0_cat)  # down-sampled, ESP_red_0

        for i, layer in enumerate(self.level2):  # p ESP-blocks, p..alpha_2
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))  # Concat_1

        output2_0 = self.level3_0(output1_cat)  # down-sampled, ESP_red_1
        for i, layer in enumerate(self.level3):  # q ESP-blocks, q..alpha_3
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))  # Concat_2

        # LSTM here
        if self.is_batch_norm:
            batch_norm_features = self.batch_norm(output2_cat)
        else:
            batch_norm_features = output2_cat
        lstm_in = batch_norm_features.view(-1, self.batch_size, batch_norm_features.shape[1],
                                           batch_norm_features.shape[2],
                                           batch_norm_features.shape[3])  # -1 ... time_steps, 1..bs
        lstm_out, new_states = self.clstm(lstm_in, states)

        classifier = F.softmax(lstm_out, dim=2)
        return classifier, new_states

    def update_parameters(self, batch_size, time_steps, overlap):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.clstm.update_parameters(batch_size, time_steps, overlap)


# ESPNet-C, Encoder part
class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, classes=16, p=2, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 19 for the cityscapes_video
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        input = input.squeeze(0)  # 5Ax -> 4Ax

        output0 = self.level1(input)  # 512x1024 --> 256x512
        inp1 = self.sample1(input)  # scale down RGB
        inp2 = self.sample2(input)  # scale down RGB

        output0_cat = self.b1(torch.cat([output0, inp1], 1))  # Concat_0
        output1_0 = self.level2_0(output0_cat)  # down-sampled, ESP_red_0

        for i, layer in enumerate(self.level2):  # p ESP-blocks, p..alpha_2
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))  # Concat_1

        output2_0 = self.level3_0(output1_cat)  # down-sampled, ESP_red_1
        for i, layer in enumerate(self.level3):  # q ESP-blocks, q..alpha_3
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))  # Concat_2

        classifier = self.classifier(output2_cat)

        classifier = F.softmax(classifier, dim=1)
        classifier = classifier.unsqueeze(0)  # 4Ax -> 5Ax
        return classifier


# Convolution with succeeding batch normalization and PReLU activation
class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        # self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


# Batch normalization with succeeding PReLU activation
class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


# Convolution with succeeding batch normalization
class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


# Convolution with zero-padding
class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


# Dilated convolution with zero-padding
class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


# ESP block with downsampling (red): Spatial dimensions /2  e.g. 256x512 -> 128x256
class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)  # os=2: difference to ESP block
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)  # convolution with different dil on
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2  # add this different dilations
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output


# ESP block: spatial dim stay the same
class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


# Apply avg-pooling n-times, RGB images with red arrow
class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, activation_function, state_init, batch_size,
                 time_steps, cell_type: int = 5, overlap: int = 0, dilation: int = 1, init: str = 'default',
                 is_stateful=True, state_img_size=None, dtype=torch.float32):
        super(ConvLSTM, self).__init__()

        self.dtype = dtype
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        if activation_function == 'tanh':
            activation_function = torch.tanh
        elif activation_function == 'lrelu':
            activation_function = F.leaky_relu
        elif activation_function == 'prelu':
            activation_function = nn.PReLU()
        self.cell_type = cell_type
        if cell_type == 5:
            self.cell = ConvLSTMCell5(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation,
                                      activation_function)
        self.is_stateful = is_stateful
        self.dtype = torch.float32
        self.device = device
        self.state_init = state_init
        self.state_img_size = state_img_size

        self.update_parameters(batch_size, time_steps, overlap)
        self.init_states(state_img_size, state_init)

        # initialization
        if init == 'default':
            self.cell.convolution.bias.data.fill_(0)  # init all biases with 0
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   0 * self.cell.hidden_channels: 1 * self.cell.hidden_channels])  # sigmoid, i
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels])  # sigmoid, f
            self.cell.convolution.bias.data[1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels].fill_(
                0.1)  # f bias
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   2 * self.cell.hidden_channels: 3 * self.cell.hidden_channels])  # sigmoid, o
            if cell_type == 5:
                nn.init.constant_(self.cell.peephole_weights, 0.1)
        if activation_function == 'tanh':
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels])  # tanh, g
        elif activation_function in ['lrelu', 'prelu']:
            nn.init.kaiming_normal_(
                self.cell.convolution.weight.data[3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels],
                nonlinearity='leaky_relu')  # lrelu, g

    def forward(self, inputs, states):  # inputs shape: time_step, batch_size, channels, height, width
        new_states = None
        time_steps = inputs.shape[0]
        outputs = torch.empty(time_steps, self.batch_size, self.hidden_channels, inputs.shape[3],
                              inputs.shape[4], dtype=self.dtype, device=device)
        if self.is_stateful == 0 or states is None:
            h = nn.functional.interpolate(self.h0.expand(self.batch_size, -1, -1, -1),
                                          size=(inputs.shape[3], inputs.shape[4]), mode='bilinear', align_corners=True)
            c = nn.functional.interpolate(self.c0.expand(self.batch_size, -1, -1, -1),
                                          size=(inputs.shape[3], inputs.shape[4]), mode='bilinear', align_corners=True)
            # print("Init LSTM")
        else:
            c = states[0]
            h = states[1]

        for time_step in range(time_steps):
            x = inputs[time_step]
            h, c = self.cell(x, h, c)  # to run hooks (pre, post) and .forward()

            if self.cell_type == 4:
                outputs[time_step] = h[:, :, 0]
            else:
                outputs[time_step] = h
            if self.is_stateful and time_step == time_steps - (self.overlap + 1):
                new_states = torch.stack((c.data, h.data))

        return outputs, new_states

    def init_states(self, state_size, state_init):
        if state_init == 'zero':
            self.h0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'rand':  # cell_state rand [0,1) init
            self.h0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'learn':
            if self.cell_type == 4:
                self.h0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype),
                                       requires_grad=True)
                self.c0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype),
                                       requires_grad=True)
            else:
                self.h0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)
                self.c0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)

    def update_parameters(self, batch_size, time_steps, overlap):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.overlap = overlap


class ConvLSTMCell5(nn.Module):  # normal conv with peephole connections
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function):
        super(ConvLSTMCell5, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o
        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels,
                                     self.kernel_size, stride=1, padding=self.padding, dilation=dilation)
        self.activation_function = activation_function
        self.peephole_weights = nn.Parameter(torch.zeros(3, self.hidden_channels), requires_grad=True)

    def forward(self, x, h, c):  # batch, channel, height, width
        x_stack_h = torch.cat((x, h), dim=1)
        A = self.convolution((x_stack_h))
        split_size = int(A.shape[1] / self.num_gates)
        (ai, af, ao, ag) = torch.split(A, split_size, dim=1)
        f = torch.sigmoid(af + c * self.peephole_weights[1, :, None, None])
        i = torch.sigmoid(ai + c * self.peephole_weights[0, :, None, None])
        g = self.activation_function(ag)
        o = torch.sigmoid(ao + c * self.peephole_weights[2, :, None, None])
        new_c = f * c + i * g
        new_h = o * self.activation_function(new_c)
        return new_h, new_c
