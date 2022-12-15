# Training Latent Graph Autoencoder

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
import os

device = 'cuda:1'

transforms = Compose([
    ToTensor(),
    Resize((320//2, 480//2)),
    ])
dataset = torchvision.datasets.ImageFolder('data/CLEVR_v1.0/images/train', transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=1)

def to_np(tnsr):
    return tnsr.detach().cpu().numpy().transpose((1,2,0))

lgvae = LatentGraphVAE(n_channels=3, h=320//2, w=480//2, device=device).to(device)
optim = torch.optim.Adam(params=lgvae.parameters())

optim.zero_grad()
batch_size = 10
n_epochs = 100
i=0
batch_loss = 0
batch_variance = 0
batch_overlap = 0
batch_l1 = 0
batch_total = 0
checkpoint = 1000
niter = 100000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs*70000//batch_size)

tmstp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')
os.mkdir(f'outputs/layerwise/{tmstp}')
os.mkdir(f'outputs/values/layerwise/{tmstp}/')

image,_ = next(iter(dataloader))

for epoch in range(n_epochs):
    i=0
    for image,_ in dataloader:
    # for j in range(niter):
        image = image.squeeze(0).to(device)
        recon, nodes, edge_attentions = lgvae(image)
        recon_loss = torch.mean((recon - image)**2) 
        loss = recon_loss
        loss.backward()
        batch_loss += float(recon_loss)/batch_size
        
        i+=1
        if i%batch_size==0:
            optim.step()
            optim.zero_grad()
            lr = scheduler.get_last_lr()
            print(f"epoch={epoch:4d} n={i:8d} loss={batch_loss:10.6f} " +
                  f"lr={lr} total={batch_total}",flush=True)
            batch_loss = 0
            # scheduler.step()
        if i%checkpoint==0:
            torch.save(lgvae.state_dict(), f'models/lgvae_{tmstp}.torch')
            torch.save(edge_attentions, f'outputs/values/layerwise/{tmstp}/attentions_{epoch}-{i}.torch')
            torch.save(nodes, f'outputs/values/layerwise/{tmstp}/nodes_{epoch}-{i}.torch')
            save_image(recon, f'outputs/layerwise/{tmstp}/layerwise_{epoch}-{i}.png')