# Training Basic (Non-Graph) Autoencoder

import torchvision
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Scale
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import one_hot
from basic_autoencoder import BasicAE
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from datetime import datetime
import os

device = 'cuda:2'

transforms = Compose([
    ToTensor(),
    Resize((320//2, 480//2)),
    ])
dataset = torchvision.datasets.ImageFolder('data/CLEVR_v1.0/images/train', transform=transforms)


def to_np(tnsr):
    return tnsr.detach().cpu().numpy().transpose((1,2,0))

ae = BasicAE(n_channels=3, w=320//2, h=480//2, device=device).to(device)
optim = torch.optim.Adam(params=ae.parameters())

dataloader = DataLoader(dataset=dataset, batch_size=1)

optim.zero_grad()
batch_size = 100
n_epochs = 50
i=0
batch_loss = 0
batch_variance = 0
batch_overlap = 0
batch_l1 = 0
batch_total = 0
checkpoint = 1000
niter = 100000

tmstp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')
os.mkdir(f'outputs/basic_ae/{tmstp}')

first_image,_ = next(iter(dataloader))
# print(image.shape)

for epoch in range(n_epochs):
    i=0
    for image,_ in dataloader:
    # for j in range(niter):
        image = image.squeeze(0).to(device)
        recon = ae(image)
        loss = torch.mean((recon - image)**2) 
        loss.backward()
        batch_loss += float(loss)/batch_size
        
        i+=1
        if i%batch_size==0:
            optim.step()
            optim.zero_grad()
            print(f"epoch={epoch:4d} n={i:8d} loss={batch_loss:8.4f} ", flush=True)
            batch_loss = 0
            # scheduler.step()
        if i%checkpoint==0:
            torch.save(ae.state_dict(), f'models/lgvae_{tmstp}.torch')
            save_image(recon, f'outputs/basic_ae/{tmstp}/basic_ae_{epoch}-{i}.png')