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

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)

        mpgg_layers = [128, 64, 32]
        # There are 4 convolutional layers, each of which reduces the size of the input images by 2
        # So, total reduction is by a factor of 2**4
        wp = w//(2**2)
        hp = h//(2**2)
        self.dim_z = 128
        self.linear_down = nn.Linear(wp*hp*32, self.dim_z)
        # TODO make this match mpgg layers in code
        self.linear_up = nn.Linear(mpgg_layers[-1], wp*hp*32)
        
        self.mpgg = MPGG(dim_z=self.dim_z, 
                           edge_dim=64, 
                           layers=mpgg_layers, 
                           max_nodes=8, device=device) #TODO maxnodes doesn't affect anything right now
        
        self.up1 = Up(32, 16)
        self.up2 = Up(16, 8)

        self.outc = OutConv(8, n_channels)

    def forward(self, x):
        x = x.unsqueeze(0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # flatten and linear projection of UNET down projection
        z = self.linear_down(x3.view(-1))
        nodes = self.mpgg(z)
        # same conv output shape as x3, but with a batch dimension for every node
        xup = self.linear_up(nodes).view((nodes.shape[0], x3.shape[1], x3.shape[2], x3.shape[3]))
        xup1 = self.up1(xup)
        xup2 = self.up2(xup1)

        xout = self.outc(xup2)
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
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU()
            )

    def forward(self, x):
        return self.conv(x)