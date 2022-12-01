from graph_autoencoder import LatentGraphVAE
import torch

device = 'cuda:1'
lgvae = LatentGraphVAE(n_channels=3, w=480, h=320, device=device).to(device)

x = torch.randn((1,480,320)).to(device)

nodes = lgvae(x)
breakpoint()