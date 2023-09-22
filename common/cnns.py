# refer to https://github.com/Oldpan/Faceswap-Deepfake-Pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        return x * y.expand_as(x)
    

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
    

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
    

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv1d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    # cols_odd = (padding_cols % 2 != 0)
    if rows_odd:
        input = F.pad(input, [0, int(rows_odd)])
    return F.conv1d(input, weight, bias, stride,
                    padding=padding_rows // 2,
                    dilation=dilation, groups=groups)

class Conv1dSamePadding(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
    

"""
from functools import reduce
from operator import __add__
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
"""


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_cols % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)

class Conv2dSamePadding(_ConvNd): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
    

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv1d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv1d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                Conv1dSamePadding(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
    

class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                Conv2dSamePadding(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
    

class SKConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1,
                 r=16, L=32, **kwargs):
        super(SKConv1d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        self.M = len(kernel_size)
        self.features = out_channels
        d = max(int(out_channels/r), L)

        self.convs = nn.ModuleList([])
        for k in kernel_size:
            self.convs.append(nn.Sequential(
                Conv1dSamePadding(
                    in_channels, out_channels, k, stride=stride,
                    padding=padding, dilation=dilation, **kwargs
                ),
                # nn.BatchNorm1d(out_ch),
                # nn.ReLU(inplace=False),
            ))
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Sequential(
            nn.Conv1d(out_channels, d, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, self.features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[-1])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        feats_Z = feats_Z.flatten(1)

        attention_vectors = [fc(feats_Z).unsqueeze_(dim=1) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1)

        fea_v = (feats * attention_vectors).sum(dim=1)

        return fea_v


class SKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1,
                 r=16, L=32, **kwargs):
        super(SKConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        self.M = len(kernel_size)
        self.features = out_channels
        d = max(int(out_channels/r), L)

        self.convs = nn.ModuleList([])
        for k in kernel_size:
            self.convs.append(nn.Sequential(
                Conv2dSamePadding(
                    in_channels, out_channels, k, stride=stride,
                    padding=padding, dilation=dilation, **kwargs
                ),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=False),
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, self.features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[-2], feats.shape[-1])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        feats_Z = feats_Z.flatten(1)

        attention_vectors = [fc(feats_Z).unsqueeze_(dim=1) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        fea_v = (feats * attention_vectors).sum(dim=1)

        return fea_v
    

if __name__ == "__main__":

    input = torch.randn(1, 3, 224)
    conv = Conv1dSamePadding(3, 64, kernel_size=7, stride=2, padding=1, dilation=3, bias=False)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112])

    input = torch.randn(1, 3, 224)
    conv = MixedConv1d(3, 64, kernel_size=[3, 5, 7], stride=2, padding=1, dilation=3, bias=False)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112])

    input = torch.randn(1, 3, 224, 224)
    conv = Conv2dSamePadding(3, 64, kernel_size=7, stride=2, padding=1, dilation=3, bias=False)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112, 112])

    input = torch.randn(1, 3, 224, 224)
    conv = MixedConv2d(3, 64, kernel_size=[3, 5, 7], stride=2, padding=1, dilation=3, bias=False)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112, 112])

    input = torch.randn(1, 3, 224)
    conv = SKConv1d(3, 64, kernel_size=[3, 5, 7], stride=2, padding=1, dilation=3, bias=False)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112])

    input = torch.randn(1, 3, 224, 224)
    conv = SKConv2d(3, 64, kernel_size=[3, 5, 7], stride=2, padding=1)
    output = conv(input)
    print(output.shape)
    # torch.Size([1, 64, 112, 112])
