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
        sample_factor = 2

        self.maxnodes = 8
        self.layers_per_node = 3
        downchannels = [32, 32, 32, 32, self.maxnodes*self.layers_per_node]
        upchannels = [self.maxnodes*self.layers_per_node, 32, 32, 32, 3]
        # There are 2 convolutional layers, each of which reduces the size of the input images by 2
        # So, total reduction is by a factor of 2**4
        wp = w//(sample_factor**(len(downchannels)-1))
        hp = h//(sample_factor**(len(downchannels)-1))
        self.wp = wp
        self.hp = hp

        self.inc = DoubleConv(n_channels, downchannels[0])
        self.downs = nn.ModuleList()
        for i in range(len(downchannels)-1):
            self.downs.append(Down(downchannels[i], downchannels[i+1], sample_factor))

        self.dim_z = wp*hp*self.layers_per_node

        mpgg_layers = [self.dim_z, self.dim_z]
        self.mpgg = MPGG(dim_z=self.dim_z, 
                         edge_dim=64, 
                         layers=mpgg_layers, 
                         max_nodes=self.maxnodes, 
                         device=device) 
        
        self.ups = nn.ModuleList()
        for i in range(len(upchannels)-1):
            self.ups.append(Up(upchannels[i], upchannels[i+1], sample_factor))

        self.outc = OutConv(upchannels[-1], n_channels)

        self.uplayers = upchannels

    def forward(self, x):
        # breakpoint()
        enc, nodes = self.encode(x)
        nodes, edge_attentions = self.graph_encode(nodes)
        out = self.decode(nodes)
        return out, nodes, edge_attentions

    def encode(self, x):
        x = x.unsqueeze(0)
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        nodes = x.view((x.shape[1]//self.layers_per_node, -1))
        return x, nodes

    def graph_encode(self, nodes, edge_index=None):
        gz, edge_attentions = self.mpgg(nodes, edge_index)
        #############################
        # TODO SOMETHING IS MESSED UP WITH THE DIMENSIONS
        # THIS SHOULD BE THE OTHER WAY (i.e. wp then hp)
        # However, either the perturbation notebook or the training breaks
        # in a mutually exclusive way when i switch them.
        #############################
        x = gz.view((1, self.maxnodes*self.layers_per_node, self.hp, self.wp)) 
        return x, edge_attentions

    def decode(self, x):
        for up in self.ups:
            x = up(x)

        xout = self.outc(x)
        return xout


""" Parts of the U-Net model """
class DoubleConv(nn.Module):
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
                