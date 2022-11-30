from graph_autoencoder import LatentGraphVAE
import torch

lgvae = LatentGraphVAE(n_channels=1, w=100, h=100)

x = torch.randn((100,100))

nodes = lgvae(x)
breakpoint()