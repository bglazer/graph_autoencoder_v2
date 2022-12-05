from graph_autoencoder import LatentGraphVAE
import torch

device = 'cuda:1'
lgvae = LatentGraphVAE(n_channels=3, w=320//2, h=480//2, device=device).to(device)

# Number of parameters in the model
# n_params = 0
# for p in lgvae.parameters():
#     if p.requires_grad:
#         print(p.shape, p.numel())
#         n_params += p.numel()
# print('-------')
# print(n_params)

# [(x.shape, x.numel()) for x in list(lgvae.parameters())]

x = torch.randn((3,320//2,480//2)).to(device)

nodes = lgvae(x)
breakpoint()