# # Training Latent Graph Autoencoder

import torchvision
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Scale
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import one_hot
from graph_autoencoder import LatentGraphVAE
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from datetime import datetime

# %%
device = 'cuda:1'

# %%
transforms = Compose([
    ToTensor(),
    Resize((320//2, 480//2)),
    ])
dataset = torchvision.datasets.ImageFolder('data/CLEVR_v1.0/images/train', transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=1)

# %%
def overlap_penalty(layers):
    nlayers = layers.shape[0]
    idxs = torch.combinations(torch.arange(nlayers)).to(device)
    # TODO this 3 assumes that there are 3 channels. Need to make this a variable
    binned = torch.clamp(layers.sum(dim=1)/3, 0)
    overlap = torch.mean((binned[idxs[:,0]] * binned[idxs[:,1]]))
    return overlap

# %%
def to_np(tnsr):
    return tnsr.detach().cpu().numpy().transpose((1,2,0))

# %%
def l2_penalty(layers):
    return torch.mean(layers**2)

# %%
def l1_penalty(layers):
    return torch.mean(torch.abs(layers))

# %%
def leaky_clamp(t, a):
    # TODO figure out how to implement this
    pass

# %%
def variance(layers):
    midx = torch.argmax(layers.sum(dim=1), dim=0)
    oh = one_hot(midx, num_classes=8).permute((2,0,1))
    return (layers.sum(dim=1) * oh).mean(dim=(1,2)).var()


# %%
def build_image(layers):
    midx = torch.argmax(layers.sum(dim=1), dim=0)
    recon = torch.gather(layers, dim=0, index=midx.expand((1,3,-1,-1))).squeeze(0)
    # recon = layers.sum(dim=0)
    return recon

lgvae = LatentGraphVAE(n_channels=3, w=320//2, h=480//2, device=device).to(device)
optim = torch.optim.Adam(params=lgvae.parameters())

optim.zero_grad()
batch_size = 100
n_epochs = 100
i=0
batch_loss = 0
batch_variance = 0
batch_overlap = 0
batch_l1 = 0
batch_total = 0
checkpoint = 1000
niter = 100000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, niter//batch_size)

tmstp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')

image,_ = next(iter(dataloader))
# print(image.shape)

for epoch in range(n_epochs):
    i=0
    for image,_ in dataloader:
    # for j in range(niter):
        image = image.squeeze(0).to(device)
        nodes = lgvae(image)
        recon = build_image(nodes)
        # recon = nodes.sum(dim=0)
        overlap = overlap_penalty(nodes)
        variance_penalty = variance(nodes)
        recon_loss = torch.mean((recon - image)**2) 
        loss = recon_loss + overlap + variance_penalty
        loss.backward()
        batch_loss += float(recon_loss)/batch_size
        batch_overlap += float(overlap)/batch_size
        batch_total += float(loss)/batch_size
        batch_variance += float(variance_penalty)/batch_size
        
        i+=1
        if i%batch_size==0:
            optim.step()
            optim.zero_grad()
            lr = scheduler.get_last_lr()
            print(f"epoch={epoch:4d} n={i:8d} loss={batch_loss:8.4f} " +
                  f"overlap={batch_overlap:8.4f} variance={batch_variance:8.4f} " + 
                  f"lr={lr} total={batch_total}",
             flush=True)
            batch_loss = 0
            batch_overlap = 0
            batch_variance = 0
            batch_l1 = 0
            batch_total = 0
            # scheduler.step()
        if i%checkpoint==0:
            torch.save(lgvae.state_dict(), f'models/lgvae_{tmstp}.torch')
            n_nodes = nodes.shape[0]
            for node_idx in range(n_nodes):
                midx = torch.argmax(nodes.sum(dim=1), dim=0)
                oh = one_hot(midx, num_classes=n_nodes)
                node = nodes[node_idx]*oh[:,:,node_idx]
                save_image(nodes[node_idx], f'outputs/graph_ae_{epoch}-{i}-{node_idx}_{tmstp}.png')
