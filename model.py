import torch
import torch.nn as nn
from layers import *
from ops import *
import torch.nn.functional as F
from preprocessing import Initializer


class GSRNet(nn.Module):

    def __init__(self, args):
        super(GSRNet, self).__init__()

        self.lr_dim = args['lr_dim']
        self.hr_dim = args['hr_dim']
        self.hidden_dim = args['hidden_dim']
        p = args['drop_p']
        ks = args['gsr_layers']

        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(self.lr_dim, self.hr_dim, ks, drop_p=p)
        self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu)

        Initializer.weights_init(self)

    def forward(self,A):

        I = torch.ones((self.lr_dim, self.lr_dim)).type(torch.FloatTensor)

        net_outs, start_gcn_outs = self.net(A, I)

        A, Z = self.layer(A, net_outs)

        hidden1 = self.gc1(Z, A)
        z = self.gc2(hidden1, A)

        z = (z + z.t())/2
        idx = torch.eye(self.hr_dim, dtype=bool) 
        z[idx]=1

        return torch.abs(z), net_outs, start_gcn_outs, A
    

class Dense(nn.Module):
    def __init__(self, n1, n2):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=0, std=0.01)

    def forward(self, x):
        out = torch.mm(x, self.weights)
        return out

class Discriminator(nn.Module):
    def __init__(self, out_dim):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(out_dim, out_dim)
        self.relu_1 = nn.ReLU(inplace=False)
        self.dense_2 = Dense(out_dim, out_dim)
        self.relu_2 = nn.ReLU(inplace=False)
        self.dense_3 = Dense(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = self.sigmoid(self.dense_3(dc_den2))
        return torch.abs(output)
    
def generate_gaussian_noise(input_layer):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=0.0, std=0.1)
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z