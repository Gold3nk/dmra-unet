from collections import OrderedDict

import torch
from torch import nn, nn as nn
from torch.nn import functional as F


def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return nn.GroupNorm(groups_, planes)


def get_norm_layer(norm_type="group"):
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None
    else:
        # return lambda x: nn.InstanceNorm3d(x, affine=True)
        return lambda x: nn.BatchNorm3d(x, affine=True)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class UBlock(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )


class UBlockCbam(nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == '__main__':
#     print(UBlock(4, 4))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


# 多尺度卷积模块
class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiConv, self).__init__()

        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.final_conv = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        # 分别对输入进行不同尺度的卷积操作
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)

        # 将不同尺度的卷积结果在通道维度上拼接起来
        out = torch.cat([out1, out3, out5], dim=1)

        # 对通道维度上的拼接结果进行1x1卷积，将通道数调整回out_channels
        out = self.final_conv(out)

        return out


# 多尺度卷积残差。先相加，再将通道数调整为输出通道数
class MultiConvRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiConvRes, self).__init__()

        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.final1_conv = nn.Conv3d(in_channels * 3, in_channels, kernel_size=1)
        self.final2_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = x
        # 分别对输入进行不同尺度的卷积操作
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)

        # 将不同尺度的卷积结果在通道维度上拼接起来
        out = torch.cat([out1, out3, out5], dim=1)

        # 对通道维度上的拼接结果进行1x1卷积，将通道数调整回in_channels
        out = self.final1_conv(out)
        # 残差连接
        out += x
        # 对通道维度上的拼接结果进行1x1卷积，将通道数调整回out_channels
        out = self.final2_conv(out)

        return out


# 多尺度卷积残差2。先将残差路径的通道数调整为输出通道数，再相加。
class MultiConvRes2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiConvRes2, self).__init__()

        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.final1_conv = nn.Conv3d(in_channels * 3, out_channels, kernel_size=1)
        self.final2_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分别对输入进行不同尺度的卷积操作
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)

        # 将不同尺度的卷积结果在通道维度上拼接起来
        out = torch.cat([out1, out3, out5], dim=1)

        # 对通道维度上的拼接结果进行1x1卷积，将通道数调整为out_channels
        out = self.final1_conv(out)
        # 对残差路径进行1x1卷积，将通道数调整为out_channels
        x = self.final2_conv(x)
        # 残差连接
        out += x

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(ChannelAttention, self).__init__()
        if out_channels is None:
            out_channels = in_channels  # 如果未提供out_channels，则将其设为与in_channels相同
        self.out_channels = out_channels  # 此行好像无用
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, out_channels),
        )
        self.sigmoid = nn.Sigmoid()

        # 如果此模块需要调整通道数，才进行此操作
        if in_channels != out_channels:
            self.x_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        # print('avg_out', avg_out.shape)  # [1, 512, 1, 1, 1]
        # print('max_out', max_out.shape)  # [1, 512, 1, 1, 1]
        avg_out = avg_out.view(avg_out.size(0), -1)  # 保持 batch_size维度 不变，而将其余维度展平为一个维度。包含通道维度
        max_out = max_out.view(max_out.size(0), -1)
        # print('avg_out', avg_out.shape)  # [1, 512]
        # print('max_out', max_out.shape)  # [1, 512]
        avg_weight = self.fc(avg_out)
        max_weight = self.fc(max_out)
        weight = avg_weight + max_weight
        weight = self.sigmoid(weight)

        # 如果此模块需要调整通道数，才进行此操作
        if hasattr(self, 'x_conv'):
            x = self.x_conv(x)

        weight = weight.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 下划线为索引值，不需要
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return x * out


class SELayer(nn.Module):
    """
    SE模块
    """
    def __init__(self, in_channels, ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

