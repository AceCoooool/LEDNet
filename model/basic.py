import torch
from torch import nn
import torch.nn.functional as F


# helper function
def channel_shuffle(x, groups):
    b, n, h, w = x.shape
    channels_per_group = n // groups

    # reshape
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(b, -1, h, w)

    return x


def basic_conv(in_channel, channel, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, channel, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
    )


# basic module
# TODO: may add bn and relu
class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(out_channel + in_channel)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu_(self.bn(x))
        return x


class SSnbtv2(nn.Module):
    def __init__(self, channel, dilate=1, drop_prob=0.01):
        super(SSnbtv2, self).__init__()
        channel = channel // 2
        self.left = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate), bias=False),
            nn.BatchNorm2d(channel), nn.Dropout2d(drop_prob, inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1), bias=False),
            nn.BatchNorm2d(channel), nn.Dropout2d(drop_prob, inplace=True)
        )

    def forward(self, x):
        x1, x2 = x.split(x.shape[1] // 2, 1)
        x1 = self.left(x1)
        x2 = self.right(x2)
        out = torch.cat([x1, x2], 1)
        x = F.relu(out + x)
        return channel_shuffle(x, 2)


class SSnbt(nn.Module):
    def __init__(self, channel, dilate=1, drop_prob=0.01):
        super(SSnbt, self).__init__()
        channel = channel // 2
        self.left = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Dropout2d(drop_prob, inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Dropout2d(drop_prob, inplace=True)
        )

    def forward(self, x):
        x1, x2 = x.split(x.shape[1] // 2, 1)
        x1 = self.left(x1)
        x2 = self.right(x2)
        out = torch.cat([x1, x2], 1)
        x = F.relu(out + x)
        return channel_shuffle(x, 2)


class APN(nn.Module):
    def __init__(self, channel, classes):
        super(APN, self).__init__()
        self.conv1 = basic_conv(channel, channel, 3, 2)
        self.conv2 = basic_conv(channel, channel, 5, 2)
        self.conv3 = basic_conv(channel, channel, 7, 2)
        self.branch1 = basic_conv(channel, classes, 1, 1)
        self.branch2 = basic_conv(channel, classes, 1, 1)
        self.branch3 = basic_conv(channel, classes, 1, 1)
        self.branch4 = basic_conv(channel, classes, 1, 1)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            basic_conv(channel, classes, 1, 1)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        out3 = self.conv1(x)
        out2 = self.conv2(out3)
        out = self.branch1(self.conv3(out2))
        out = F.interpolate(out, size=((h + 3) // 4, (w + 3) // 4), mode='bilinear', align_corners=True)
        out = out + self.branch2(out2)
        out = F.interpolate(out, size=((h + 1) // 2, (w + 1) // 2), mode='bilinear', align_corners=True)
        out = out + self.branch3(out3)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        out = out * self.branch4(x)
        out = out + self.branch5(x)
        return out


if __name__ == '__main__':
    # model = DownSampling(32)
    # a = torch.randn(1, 32, 512, 256)
    # out = model(a)
    # print(out.shape)

    # model = SSnbt(10, 2)
    # a = torch.randn(1, 20, 10, 10)
    # out = model(a)
    # print(out.shape)
    # model = basic_conv(10, 20, 3, 2)
    # a = torch.randn(1, 10, 128, 65)
    # out = model(a)
    # print(out.shape)

    model = APN(64, 10)
    x = torch.randn(2, 64, 127, 65)
    out = model(x)
    print(out.shape)
