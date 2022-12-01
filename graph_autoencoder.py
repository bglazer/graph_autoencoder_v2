import torch
import torch_geometric

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpgg import MPGG

# The image encoder is a modified version of a UNET implementation taken from:
# https://github.com/milesial/Pytorch-UNet

class LatentGraphVAE(nn.Module):
    def __init__(self, n_channels, w, h, device):
        super(LatentGraphVAE, self).__init__()
        self.device = device
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # There are 4 convolutional layers, each of which reduces the size of the input images by 2
        # So, total reduction is by a factor of 2**4
        wp = w//(2**4)
        hp = h//(2**4)
        self.linear_down = nn.Linear(wp*hp*1024, 1024)
        # TODO make this match mpgg layers in code
        self.linear_up = nn.Linear(256, wp*hp*1024)
        
        self.mpgg = MPGG(dim_z=1024, 
                           edge_dim=128, 
                           layers=[1024, 512, 256], 
                           max_nodes=8, device=device) #TODO maxnodes doesn't affect anything right now
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x = x.unsqueeze(0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # flatten and linear projection of UNET down projection
        z = self.linear_down(x5.view(-1))
        nodes = self.mpgg(z)
        # same conv output shape as x5, but with a batch dimension for every node
        xup = self.linear_up(nodes).view((-1, x5.shape[1], x5.shape[2], x5.shape[3]))
        xup1 = self.up1(xup)
        xup2 = self.up2(xup1)
        xup3 = self.up3(xup2)
        xup4 = self.up4(xup3)
        xout = self.outc(xup4)
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
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        # x = torch.cat(x1, dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)