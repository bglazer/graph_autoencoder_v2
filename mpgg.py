from torch_geometric import nn
from edge_conv import EdgeConv
import torch_geometric as pyg
import torch
from torch import relu
from torch.nn import Linear, ReLU, Sigmoid, Sequential, ModuleList
from torch_geometric.data.data import Data

def MLP(layers):
    mlp = []
    for i in range(len(layers)-2):
        mlp += [
            Linear(layers[i], layers[i+1]),
            ReLU()
        ]
    mlp.append(Linear(layers[-2], layers[-1]))
    return Sequential(*mlp)
        
# Message Passing Graph Generator
class MPGG(torch.nn.Module):
    def __init__(self, dim_z, edge_dim, layers, layers_per_node, max_nodes, device):
        super(MPGG, self).__init__()
        self.device = device
        self.dim_z = dim_z
        self.edge_dim = edge_dim
        self.layers_per_node = layers_per_node
        
        self.attentions = ModuleList()
        for i in range(len(layers)):
            self.attentions.append(MLP([edge_dim+layers[i], edge_dim//2, 1]))
        
        self.convs = ModuleList()
        for i in range(len(layers)-1):
            self.convs.append(EdgeConv(layers[i], layers[i+1], nn=self.attentions[i]))
        
        self.max_nodes = max_nodes
        self.edge_generator = MLP([layers[0]*2, edge_dim*2, edge_dim])
        
    def forward(self, z):
        nodes = z.view((z.shape[1]//self.layers_per_node, -1))

        # Create the fully connected edge index
        # combinations creates edge index without self loops
        # TODO modify this to work with varying numbers of nodes
        a = torch.combinations(torch.arange(self.max_nodes), r=2)
        # exchanges cols of "a", i.e. create edges in opposite direction
        b = a[:,[1,0]]
        edge_index = torch.vstack((a,b)).t().to(self.device)
        # Generate edge attributes for all pairs of nodes
        srcs = nodes[edge_index[0,:]]
        dsts = nodes[edge_index[1,:]]
        nodepairs = torch.cat((srcs,dsts), dim=1)
        edges = self.edge_generator(nodepairs)
        
        # TODO apply edge attention here??
        # TODO graph batching

        # Run convolutions
        for conv in self.convs[:-1]:
            nodes,_ = conv(nodes, edge_index, edges)
            nodes = relu(nodes)
        nodes, attentions = self.convs[-1](nodes, edge_index, edges)
        
        # output = torch.sum(nodes, dim=0)
        
        return nodes, attentions
