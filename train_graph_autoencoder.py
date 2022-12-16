# Training Latent Graph Autoencoder

from datetime import datetime
tmstp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M')

import torchvision
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import GaussianBlur, ColorJitter, RandomErasing
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import one_hot
from graph_autoencoder import LatentGraphVAE
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os
import sys

quiet = False
if len(sys.argv) > 2:
    if sys.argv[2] == 'q':
        quiet = True

device = sys.argv[1]

transforms = Compose([
    ToTensor(),
    Resize((320//2, 480//2)),
    ])

dataset = torchvision.datasets.ImageFolder('data/CLEVR_v1.0/images/train', transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=1)

def to_np(tnsr):
    return tnsr.detach().cpu().numpy().transpose((1,2,0))

lgvae = LatentGraphVAE(n_channels=3, h=320//2, w=480//2, device=device).to(device)
optim = torch.optim.Adam(params=lgvae.parameters(), lr=.001/5)

optim.zero_grad()
batch_size = 10
n_epochs = 100
i=0
batch_loss = 0
batch_contrast = 0
batch_total = 0
checkpoint = 1000
niter = 100000
margin = 1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs*70000//batch_size)

if not quiet:
    os.mkdir(f'outputs/contrastive/{tmstp}')
    os.mkdir(f'outputs/values/contrastive/{tmstp}/')

image,_ = next(iter(dataloader))

# Triplet loss with positive and negative examples
def contrastive_loss(embedding, positives, negatives, margin=1):
    positives = torch.mean((embedding - positives)**2)
    negatives = torch.mean((embedding - negatives)**2)
    loss = positives - negatives + margin
    return torch.max(loss, torch.zeros_like(loss))

def generate_negative(dataset):
    # Generate n random integers
    r = np.random.randint(0, len(dataset))
    # Get the images
    rand_image = dataset[r][0]
    rand_image = rand_image.squeeze(0).to(device)
    return rand_image

augmentations = [
    # RandomPosterize(),
    GaussianBlur(11, sigma=(1.0, 100.0)),
    ColorJitter(.3),
    RandomErasing(p=1.0, value=(.5,.5,.5), scale=[.01,.1])
    # RandomPerspective(),
    # RandomAdjustSharpness(3)
]

def generate_positive(image, augmentations):
    # Select a number of augmentations
    n_aug = np.random.randint(1, len(augmentations))
    # Randomly sample n augmentations
    random_augs = np.random.choice(augmentations, n_aug)
    # Apply them to the image
    for aug in random_augs:
        image = aug(image)

    return image

for epoch in range(n_epochs):
    i=0
    for image,_ in dataloader:
        image = image.squeeze(0).to(device)
        recon, nodes, edge_attentions = lgvae(image)
        recon_loss = torch.mean((recon - image)**2) 
        positive = generate_positive(image, augmentations)
        pos_recon, pos_nodes, pos_edges = lgvae(positive)
        negative = generate_negative(dataset)
        neg_recon, neg_nodes, neg_edges = lgvae(negative)
        # TODO - should I have recon loss for positives also?
        # TODO - contrast loss for edges as well
        # TODO - ordered contrast loss on perturbed graph in image space
        # TODO - maybe clamp embeddings to pointwise [0-1] (i.e. L_inf=1)
        #      - or maybe just [0-..] ?
        contrast_loss = contrastive_loss(nodes, pos_nodes, neg_nodes, margin=margin)
        loss = recon_loss + contrast_loss
        loss.backward()
        batch_loss += float(recon_loss)/batch_size
        batch_contrast += float(contrast_loss)/batch_size
        batch_total += float(loss)/batch_size
        
        i+=1
        if i%batch_size==0:
            optim.step()
            optim.zero_grad()
            lr = scheduler.get_last_lr()
            print(f"epoch={epoch:4d} n={i:8d} recon_loss={batch_loss:10.6f} " +
                  f"contrast_loss={batch_contrast:10.6f} total={batch_total:10.6f} " +
                  f"lr={lr} total={batch_total}",flush=True)
            batch_loss = 0
            batch_contrast = 0
            batch_total = 0
            # scheduler.step()
        if i%checkpoint==0:
            if not quiet:
                torch.save(lgvae.state_dict(), f'models/lgvae_{tmstp}.torch')
                torch.save(edge_attentions, f'outputs/values/contrastive/{tmstp}/attentions_{epoch}-{i}.torch')
                torch.save(nodes, f'outputs/values/contrastive/{tmstp}/nodes_{epoch}-{i}.torch')
                save_image(recon, f'outputs/contrastive/{tmstp}/contrastive_{epoch}-{i}.png')