# # Perturbing graph structure

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

def to_np(tnsr):
    return tnsr.squeeze(0).detach().cpu().numpy().transpose((1,2,0))

device='cuda:3'

sd = torch.load(open('models/lgvae_20221207-2032.torch','rb'), map_location=device)

lgvae = LatentGraphVAE(n_channels=3, w=480//2, h=320//2, device=device)

lgvae.load_state_dict(sd)

lgvae = lgvae.to(device)

transforms = Compose([
    ToTensor(),
    Resize((320//2, 480//2)),
    ])
dataset = torchvision.datasets.ImageFolder('data/CLEVR_v1.0/images/train', transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=1)

idl = iter(dataloader)
image,_ = next(idl)
image = image.to(device)

recon, nodes, attentions = lgvae(image.squeeze(0))
plt.axis('off')
plt.imshow(to_np(recon))

max_nodes = 8
layers_per_node = 3
nodes = nodes.view((max_nodes,-1))

a = torch.combinations(torch.arange(max_nodes), r=2)
b = a[:,[1,0]]
edge_index = torch.vstack((a,b)).t().to(device)

def remove_node(nodes, edge_index, node_idx):
    new_edges = edge_index[:,(edge_index != node_idx).sum(dim=0) == 2]
    new_nodes_mask = torch.arange(nodes.shape[0]) == node_idx
    nodes[new_nodes_mask]=0.0
    return nodes, new_edges

n,e = remove_node(nodes, edge_index, 3)
perturbed = lgvae.graph_encode(n, e)
plt.imshow(to_np(perturbed))