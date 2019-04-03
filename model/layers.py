import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear(in_channels, out_channels, use_bias=True):
    return nn.Linear(in_channels, out_channels, bias=use_bias)

def conv(in_planes,  out_planes, kernel_size, stride=1, padding=0, use_bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)

def conv_transpose(in_planes,  out_planes, kernel_size, stride=1, padding=0, use_bias=True):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)

def conv1x1(in_planes,  out_planes, stride=1, use_bias=True):
    return conv(in_planes, out_planes, 1, stride=stride, padding=0, use_bias=use_bias)

def conv3x3(in_planes,  out_planes, stride=1, use_bias=True):
    return conv(in_planes, out_planes, 3, stride=stride, padding=1, use_bias=use_bias)

def conv5x5(in_planes,  out_planes, stride=1, use_bias=True):
    return conv(in_planes, out_planes, 5, stride=stride, padding=2, use_bias=use_bias)

def conv7x7(in_planes,  out_planes, stride=1, use_bias=True):
    return conv(in_planes, out_planes, 7, stride=stride, padding=3, use_bias=use_bias)

def convTranspose2x2(in_planes, out_planes, stride=2, use_bias=True):
    return conv_transpose(in_planes, out_planes, 2, stride=stride, padding=0, use_bias=use_bias)

def convTranspose4x4(in_planes, out_planes, stride=2, use_bias=True):
    return conv_transpose(in_planes, out_planes, 4, stride=stride, padding=1, use_bias=use_bias)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fcn=None, dropout=False, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fcn = activation_fcn
        self.batch_norm = batch_norm

        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            use_bias=False if batch_norm else True)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.actv = activation_fcn if activation_fcn is not None else None
        self.dropout = nn.Dropout2d(p=0.5) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.actv is not None:
            x = self.actv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class BasicResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False, dropout=False):
        super(BasicResBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.downsample = downsample
        self.dropout = dropout

        # Conv1
        if downsample:
            self.conv1 = conv3x3(in_planes, out_planes, stride=2, use_bias=False)
        else:
            self.conv1 = conv3x3(in_planes, out_planes, stride=1, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # Conv2
        self.conv2 = conv3x3(out_planes, out_planes, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # ConvRes
        if downsample:
            self.conv_res = conv1x1(in_planes, out_planes, stride=2, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_planes)
        elif in_planes != out_planes:
            self.conv_res = conv1x1(in_planes, out_planes, stride=1, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_planes)
        else:
            self.conv_res = None
            self.bn_res = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.dropout:
            y = F.dropout2d(y, 0.5, self.training, True)
        res = x if self.conv_res is None else self.bn_res(self.conv_res(x))
        return y + res


class BottleneckResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=None, dropout=False):
        super(BottleneckResBlock, self).__init__()
        self.dropout = dropout

        # Conv1
        self.conv1 = conv1x1(in_planes, out_planes//4, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes//4)

        # Conv2
        if downsample:
            self.conv2 = conv3x3(out_planes//4, out_planes//4, stride=2, use_bias=False)
        else:
            self.conv2 = conv3x3(out_planes//4, out_planes//4, stride=1, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes//4)

        # Conv3
        self.conv3 = conv1x1(out_planes//4, out_planes, use_bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # ConvRes
        if downsample:
            self.conv_res = conv1x1(in_planes, out_planes, stride=2, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_planes)
        elif in_planes != out_planes:
            self.conv_res = conv1x1(in_planes, out_planes, stride=1, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_planes)
        else:
            self.conv_res = None
            self.bn_res = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.dropout:
            y = F.dropout2d(y, 0.5, self.training, True)
        res = x if self.conv_res is None else self.bn_res(self.conv_res(x))
        return F.relu(y + res)