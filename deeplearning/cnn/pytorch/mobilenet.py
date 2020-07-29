# -*- coding: utf-8 -*-
#
# reference:
# https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
# https://blog.csdn.net/winycg/article/details/86662347
# https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# https://blog.csdn.net/winycg/article/details/87474788

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size,
                      stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DepthWise(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DepthWise, self).__init__(
            # depthwise
            ConvBNReLU(in_planes, in_planes, kernel_size=3,
                           stride=stride, groups=in_planes),
            # pointwise
            ConvBNReLU(in_planes, out_planes, kernel_size=1)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_planes = int(in_planes * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(
                ConvBNReLU(in_planes, hidden_planes, kernel_size=1)
            )
        layers.extend([
            # depthwise
            ConvBNReLU(hidden_planes, hidden_planes, kernel_size=3,
                           stride=stride, groups=hidden_planes),
            # pointwise-linear
            nn.Conv2d(hidden_planes, out_planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes)
            # no ReLU here
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet_v1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet_v1, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, kernel_size=3)
        self.features = self._make_layers(in_planes=32)
        self.classifer = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(DepthWise(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


class MobileNet_v2(nn.Module):
    cfg = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2], # NOTE: change stride 2 -> 1 for CIFAR10
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, num_classes=10):
        super(MobileNet_v2, self).__init__()

        in_planes = 32
        out_planes = 1280
        # building first layer
        self.conv1 = ConvBNReLU(3, 32, kernel_size=3)
        # building inverted residual blocks
        self.features = self._make_layers(in_planes, out_planes)
        # building classifier
        self.classifier = nn.Linear(out_planes, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, out_planes):
        layers = []
        for expand_ratio, outp, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(in_planes, outp, stride, expand_ratio))
                in_planes = outp
        layers.append(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                      padding=0, bias=False)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":

    net = MobileNet_v2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
