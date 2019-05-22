from torch import nn
import torch.nn.functional as F
from model.basic import DownSampling, SSnbt, APN


class LEDNet(nn.Module):
    def __init__(self, nclass, drop=0.1):
        super(LEDNet, self).__init__()
        self.encoder = nn.Sequential(
            DownSampling(3, 29), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop),
            DownSampling(32, 32), SSnbt(64, 1, 0.1 * drop), SSnbt(64, 1, 0.1 * drop),
            DownSampling(64, 64), SSnbt(128, 1, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop),
            SSnbt(128, 9, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop), SSnbt(128, 9, drop), SSnbt(128, 17, drop)
        )
        self.decoder = APN(128, nclass)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


if __name__ == '__main__':
    net = LEDNet(21)
    import torch

    a = torch.randn(2, 3, 554, 253)
    out = net(a)
    print(out.shape)
