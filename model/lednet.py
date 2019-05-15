from torch import nn
import torch.nn.functional as F
from model.basic import DownSampling, SSnbt, APN


class LEDNet(nn.Module):
    def __init__(self, nclass):
        super(LEDNet, self).__init__()
        self.conv = nn.Sequential(  # not sure
            nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            DownSampling(16), SSnbt(32), SSnbt(32), SSnbt(32),
            DownSampling(32), SSnbt(64), SSnbt(64),
            DownSampling(64), SSnbt(128, 1), SSnbt(128, 2), SSnbt(128, 5),
            SSnbt(128, 9), SSnbt(128, 2), SSnbt(128, 5), SSnbt(128, 9), SSnbt(128, 17)
        )
        self.decoder = APN(128, nclass)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


if __name__ == '__main__':
    net = LEDNet(21)
    import torch
    a = torch.randn(2, 3, 1024, 512)
    out = net(a)
    print(out.shape)