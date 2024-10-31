import torch
import torch.nn as nn
import numpy as np
from preprocessing import normalize_adj_torch


class GSRLayer(nn.Module):
  
    def __init__(self,hr_dim):
        super(GSRLayer, self).__init__()

        self.hr_dim = hr_dim

        # Xavier Glorot's initialisation
        r = np.sqrt(6.0 / (hr_dim + hr_dim)) 
        self.weights = torch.nn.Parameter(data=2*r*torch.randn((hr_dim, hr_dim)) + r, requires_grad = True)

    def forward(self,A,X):
        lr = A
        lr_dim = lr.shape[0]
        f = X
        _, U_lr = torch.linalg.eigh(lr, UPLO='U')
        
        eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
        s_d = torch.cat((eye_mat,eye_mat),0)
        
        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a, torch.t(U_lr))
        f_d = torch.matmul(b ,f)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)
        adj = normalize_adj_torch(self.f_d)
        X = torch.mm(adj, adj.t())
        X = (X + X.t())/2
        idx = torch.eye(self.hr_dim, dtype=bool)
        X[idx]=1
        return adj, torch.abs(X)