import torch.nn as nn
import torch.nn.functional as F

from mpgg import MPGG

# The image encoder is a modified version of a UNET implementation taken from:
# https://github.com/milesial/Pytorch-UNet

class BasicAE(nn.Module):
    def __init__(self, n_channels, w, h, device):
        super(BasicAE, self).__init__()
        self.device = device
        self.n_channels = n_channels
        sample_factor = 2

        downlayers = [32, 32, 32, 32]
        uplayers = [32, 32, 32, 3]
        # There are N convolutional layers, each of which reduces the size of the input images by sample_factor
        # So, total reduction is by a factor of N**sample_factor-1
        wp = w//(sample_factor**(len(downlayers)-1))
        hp = h//(sample_factor**(len(downlayers)-1))

        self.inc = DoubleConv(n_channels, downlayers[0])
        self.downs = nn.ModuleList()
        for i in range(len(downlayers)-1):
            self.downs.append(Down(downlayers[i], downlayers[i+1], sample_factor))

        self.dim_z = wp*hp
        # self.linear_down = nn.Linear(wp*hp, self.dim_z)
        
        # self.linear_up = nn.Linear(self.dim_z, wp*hp*uplayers[0])
        self.ups = nn.ModuleList()
        for i in range(len(uplayers)-1):
            self.ups.append(Up(uplayers[i], uplayers[i+1], sample_factor))

        self.outc = OutConv(uplayers[-1], n_channels)

        self.uplayers = uplayers

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.inc(x)
        for down in self.downs:
            x = down(x)

        # x = x.amax(dim=1).unsqueeze(1)
        # z = self.linear_down(xflat.view(-1))
        # x = self.linear_up(z).view((1,self.uplayers[0],xflat.shape[2],xflat.shape[3]))

        for up in self.ups:
            x = up(x)

        xout = self.outc(x)
        return xout
        # return logits


""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, sample_factor):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(sample_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, sample_factor):
        super().__init__()

        self.up = nn.Upsample(scale_factor=sample_factor, mode='bilinear', align_corners=True)
        # TODO maybe change back to //2
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU()
            # nn.Sigmoid()
            )

    def forward(self, x):
        return self.conv(x)