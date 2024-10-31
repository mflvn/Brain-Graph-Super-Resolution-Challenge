import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class GraphUnpool(nn.Module):
    def __init__(self, *args):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X

    
class GraphPool(nn.Module):
    def __init__(self, in_dim, k, p = 0):
        super(GraphPool, self).__init__()
        self.k = k
        self.W = GAT(in_dim, 1)
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, A, X):
        Z = self.drop(X)
        scores = self.W(A, Z).squeeze() / 100

        scores = F.sigmoid(scores)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))

        new_X = X[idx, :]
        new_X = torch.mul(new_X,  torch.unsqueeze(values, -1))
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GCN(nn.Module):
    '''
    Defines a GCN layer as a linear layer between `in_dim` and `out_dim`, with a 
    dropout probability `p` and a configurable activation function `act`.
    '''
    def __init__(self, in_dim, out_dim, act = F.relu, p = 0):
        super(GCN, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act = act

    def forward(self, A, X):
        X = self.drop(X)
        X = self.W(X)
        # X = torch.matmul(A, X)
        return self.act(X)


class GraphUnet(nn.Module):

    def __init__(self, in_dim, out_dim, ks, dim=320, drop_p = 0):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.l_n = len(ks)

        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.attentions = []
        
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, p = drop_p))
            self.up_gcns.append(GCN(2*dim, dim, p = drop_p))
            self.pools.append(GraphPool(dim, ks[i], p = drop_p))
            self.unpools.append(GraphUnpool())
            self.attentions.append(AttentionGate(dim, dim))

    def forward(self, A, X):
        adjs, idxs, down_outs = [], [], []

        X = self.start_gcn(A, X)
        start_gcn_outs = X

        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adjs.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            idxs.append(idx)

        X = self.bottom_gcn(A, X)

        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
           
            A, idx = adjs[up_idx], idxs[up_idx]
            A, X = self.unpools[i](A, X, idx)

            X = torch.cat([X, self.attentions[i](X, down_outs[up_idx])], 1)

            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        
        X = torch.cat([X, start_gcn_outs], 1)
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs
    
class GraphConvolution(nn.Module):
    """
    GCN layer with residual connection
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.w1 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.w2 = nn.Parameter(torch.FloatTensor(out_features, out_features))

        self.skip_transform = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        nn.init.xavier_uniform_(self.skip_transform.weight)
        nn.init.zeros_(self.skip_transform.bias)

    def forward(self, A, X):
        conv_output = self.act(A @ (X @ self.w1))
        conv_output = self.act(A @ (conv_output @ self.w2))

        conv_output = F.dropout(conv_output, self.dropout, self.training)
        skip_connection = A @ self.skip_transform(X)
        output = conv_output + skip_connection
        return self.act(output)



class AttentionGate(nn.Module):

    def __init__(self, in_dim, out_dim, act = F.relu) -> None:
        super(AttentionGate, self).__init__()

        self.Wg = nn.Linear(in_dim, out_dim)
        self.Ws = nn.Linear(in_dim, out_dim)
        self.act = act

        self.psi = nn.Linear(out_dim, out_dim)

    def forward(self, X, Xp):

        Wg = self.Wg(X)
        Ws = self.Ws(Xp)

        out = self.act(Wg + Ws)
        out = F.sigmoid(self.psi(out))

        return out
    
class GAT(nn.Module):
    def __init__(self, in_features, out_features, activation = None):
        super(GAT, self).__init__()
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation

    def forward(self, A, X):
        N = X.shape[0]
        Hp = self.weight(X) + self.bias
        S = torch.zeros((N, N))
        
        a, b = Hp.shape

        expanded_tensor = Hp.repeat(a, 1)
        tiled_tensor = Hp.repeat(1, a).view(-1, b)
        output_tensor = torch.cat((expanded_tensor, tiled_tensor), dim=1)   

        S = (output_tensor @ self.phi).reshape((N,N))
        
        mask = A + torch.eye(A.shape[1])
        S_masked = F.softmax(torch.where(mask != 0, S, -1e7), dim=1)

        h = S_masked @ Hp 

        return self.activation(h) if self.activation else h
