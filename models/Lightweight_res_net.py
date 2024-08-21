"""
Author:  Qi tong Chen
Date: 2024.08.20
Model
"""
import torch
import torch.nn as nn
import math
from models.Lightweight_res_blo import InvertedResidual_Block
from torchstat import stat
from ptflops import get_model_complexity_info
from utils_dsbn.train_utils import init_weights

from Attention.CloAttention import EfficientAttention


class Lightweight_model(nn.Module):
    def __init__(self, input_size=32, class_num=4):
        """
        Diagnostic model
        :param class_num: Number of health states
        """
        super(Lightweight_model, self).__init__()
        model_size = '0.5x'
        print('model size is ', model_size)
        self.stage_repeats = [1, 1]
        self.expand_ratio = 1
        self.model_size = model_size
        self.n_class = class_num

        self.avegpool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        if model_size == '0.5x':
            self.stage_out_channels = [-1, 48, 96, 192]

        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, input_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )
        # 2. Two modules
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                # Each module first performs downsampling with stride=2
                if idxstage == 0 and i == 0:
                    self.features.append(self.first_conv)
                if i == 0:
                    self.features.append(InvertedResidual_Block(inp=input_channel, oup=input_channel, stride=2,
                                                                expand_ratio=3))
                else:
                    self.features.append(InvertedResidual_Block(inp=input_channel, oup=input_channel, stride=1,
                                                                expand_ratio=3))
                input_channel = output_channel

        self.feature_extractor = nn.Sequential(*self.features)

        self.clf1 = nn.Sequential(
            # GC
            nn.Conv2d(in_channels=self.stage_out_channels[-1], out_channels=96,
                      kernel_size=3, stride=1, padding=1, groups=96),
            # nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=96),
            nn.LeakyReLU(0.2),
        )
        self.clf2 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, stride=2, padding=1, groups=96),
            nn.BatchNorm2d(num_features=96),
            nn.LeakyReLU(0.2),
        )
        self.clf3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=self.n_class, kernel_size=1, stride=2, padding=0, groups=1),
            # nn.Dropout(0.2),
        )
        self.block1 = nn.Sequential(EfficientAttention(192))
        self.block2 = nn.Sequential(EfficientAttention(96))
        self._initialize_weights()

    def dim_show(self, input):
        """
        The output dimension of each layer of the test model
        :param input: The size of the feature
        :return:
        """
        X = input

        print('1.Feature extractor：')
        for layer in self.feature_extractor:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        for layer in self.clf1:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        for layer in self.clf2:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        for layer in self.clf3:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('2.Output：')

    def forward(self, x):
        x1 = self.feature_extractor(x)
        # x1 = self.block1(x1)
        x2 = self.clf1(x1)
        x2 = self.block2(x2)
        x3 = self.clf2(x2)
        x4 = self.clf3(x3)
        x5 = x4.contiguous().view(-1, self.n_class)

        return x5, x4, x3, x2, x1

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        x = torch.cat((x[0], x[1]), 1)
        return x

    # Adjust the weights and bias parameters of each layer of the model
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Group Discriminate
class Dmodel(nn.Module):
    def __init__(self, output_groups=16):
        """
        Group Discriminate
        Args:
            output_groups：Number of groups
        """
        super(Dmodel, self).__init__()
        self.output_groups = output_groups
        self.Dclf = nn.Sequential(
            nn.Conv2d(4*2, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, self.output_groups, 1, 1, 0),
        )
        init_weights(self)
        last_layer = self.Dclf[6]
        last_layer.weight.data.normal_(0, 0.3).clamp_(min=-0.6, max=0.6)
        last_layer.bias.data.zero_()

    def forward(self, dx):
        """
        Args:
            dx:Input data
        return：
            Group labels
        """
        dx = self.Dclf(dx)
        return dx.view(dx.size(0), dx.size(1))


class Domain_model(nn.Module):
    def __init__(self, output_domain=1):
        """
        Domain
        Args:
            output_domain
        """
        super(Domain_model, self).__init__()
        self.output_domain = output_domain
        self.Dclf = nn.Sequential(

            nn.Conv2d(4, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, self.output_domain, 1, 1, 0),

        )

        init_weights(self)
        last_layer = self.Dclf[6]
        last_layer.weight.data.normal_(0, 0.3).clamp_(min=-0.6, max=0.6)
        last_layer.bias.data.zero_()

    def forward(self, dx):
        """
        Forward Propagation
        return：
            Domain labels
        """

        dx = self.Dclf(dx)
        return dx.view(dx.size(0), dx.size(1))


if __name__ == "__main__":
    model = Lightweight_model(class_num=4)
    # print(model)

    test_data_1 = torch.rand(64, 1, 32, 32)
    test_data_2 = torch.rand(64, 1, 32, 32)
    model.dim_show(test_data_1)
    test_outputs_1 = model(test_data_1)
    cls_out_1, x4, x3, x2, x1 = test_outputs_1
    print('Classifier output：{}'.format(cls_out_1.size()))
    print('Classifier output：{}，x4：{}, x3：{}, x2：{}, x1：{}'.format(cls_out_1.size(), x4.size(), x3.size(), x2.size(), x1.size()))

    # stat(model, (1, 32, 32))

    cls_out_2, x4_2, x3_2, x2_2, x1_2 = model(test_data_2)
    fea = torch.cat((x4, x4_2), dim=1)
    dmodel = Dmodel(output_groups=16)
    dout = dmodel.forward(fea)
    print('判别器：dout.shape = ', dout.shape)
