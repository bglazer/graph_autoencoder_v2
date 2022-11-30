from mpgg import MPGG
import torch

mpgg = MPGG(dim_z=10, edge_dim=5, layers=[10,9,8], max_nodes=4)
z = torch.randn((10))
nodes = mpgg(z)
print(nodes)