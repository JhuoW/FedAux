import torch.nn as nn
from torch.nn import functional as F, init
from misc.utils import *

# acc: 82.06 on 10 client Cora
# class FedAux(nn.Module):
#     def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
#         super().__init__()
#         self.n_feat = n_feat
#         self.n_dims = n_dims
#         self.n_clss = n_clss
#         self.args = args
#         self.sigma = args.sigma
#         from models.layers import MaskedGCNConv, MaskedLinear
#         self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
#         self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
#         # from torch_geometric.nn import GATConv, GCNConv
#         # self.conv1 = GCNConv(self.n_feat, self.n_dims)
#         # self.conv2 = GCNConv(self.n_dims, self.n_dims)

#         self.clsif = MaskedLinear(2 * self.n_dims, self.n_clss, l1=l1, args=args)
#         # self.clsif = nn.Linear(2 * self.n_dims, self.n_clss)
       
#         #####
#         # self.clsif1 = MaskedLinear(2 * self.n_dims, self.n_dims, l1=l1, args=args)
#         # self.clsif2 = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)
#         #####

#         anchor_dim = n_dims
#         self.aux   = nn.Parameter(torch.empty(anchor_dim))
#         init.uniform_(self.aux)

#     def _kernel_aggregate(self, h: torch.Tensor, edge_index) -> torch.Tensor:
#         """
#         h : [N, d]           node embeddings
#         returns z : [N, d]   kernel‑smoothed embeddings
#         """
#         # a = F.normalize(self.aux, dim=0)                          # unit vector
#         score = F.cosine_similarity(h, self.aux.unsqueeze(0), dim=-1)    # s_i  (N)
#         diff  = score.unsqueeze(0) - score.unsqueeze(1)           # (N,N)
#         kappa = torch.exp(-(diff ** 2) / (self.sigma ** 2))       # eq.(3)
#         z     = (kappa @ h) / kappa.sum(dim=1, keepdim=True)
#         return z
    
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         z = self._kernel_aggregate(x, edge_index)
#         #####
#         x = self.clsif(torch.cat([x, z], dim=-1))
#         #####

#         ###
#         # x = self.clsif1(torch.cat([x, z], dim=-1))
#         # x = F.relu(x)
#         # x = F.dropout(x, training=self.training)
#         # x = self.clsif2(x)
#         ###
#         return x



class FedAux(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args
        self.sigma = args.sigma
        self.dropout1 = args.dropout1
        # from models.layers import MaskedGCNConv, MaskedLinear
        # self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        # self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv, APPNP
        self.conv1 = SAGEConv(self.n_feat, self.n_dims)
        self.conv2 = SAGEConv(self.n_dims, self.n_dims)
        # self.clsif = MaskedLinear(2 * self.n_dims, self.n_clss, l1=l1, args=args)
        self.clsif = nn.Linear(2 * self.n_dims, self.n_clss)
       
        #####
        # self.clsif1 = MaskedLinear(2 * self.n_dims, self.n_dims, l1=l1, args=args)
        # self.clsif2 = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)
        #####

        anchor_dim = n_dims
        self.aux   = nn.Parameter(torch.empty(anchor_dim))
        init.uniform_(self.aux)

    def _kernel_aggregate(self, h: torch.Tensor, edge_index) -> torch.Tensor:
        """
        h : [N, d]            node embeddings
        returns z : [N, d]    kernel‑smoothed embeddings
        """
        # a = F.normalize(self.aux, dim=0)                          # unit vector
        score = F.cosine_similarity(h, self.aux.unsqueeze(0), dim=-1)    # s_i  (N)
        diff  = score.unsqueeze(0) - score.unsqueeze(1)           # (N,N)
        # using edge_index to sparsify the diff matrix
        mask[edge_index[0], edge_index[1]] = diff[edge_index[0], edge_index[1]]
        sparse_diff = mask.to_sparse().coalesce()
        kappa = torch.exp(-(diff ** 2) / (self.sigma ** 2))       # eq.(3)
        z     = (kappa @ h) / kappa.sum(dim=1, keepdim=True)
        return z
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout1, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        z = self._kernel_aggregate(x, edge_index)
        #####
        x = self.clsif(torch.cat([x, z], dim=-1))
        #####

        ###
        # x = self.clsif1(torch.cat([x, z], dim=-1))
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.clsif2(x)
        ###
        return x











class GNNAUX(nn.Module):
    def __init__(self, n_feat, n_dims, n_clss, args=None):
        super().__init__()
        from torch_geometric.nn import GCNConv, SAGEConv
        self.conv1 = SAGEConv(n_feat, n_dims)
        self.conv2 = SAGEConv(n_dims, n_dims)
        anchor_dim = n_dims
        self.aux   = nn.Parameter(torch.empty(anchor_dim))  #  \boldsymbol a
        init.normal_(self.aux)
        self.clsif   = nn.Linear(2 * n_dims, n_clss)
        # self.clsif   = nn.Linear(n_dims, n_clss)
        self.sigma = args.sigma
        # self.sigma = nn.Parameter(torch.randn(1))
        self.dropout1 = args.dropout1
        # self.reset_parameters()


    # ---------- kernel‑aggregator ------------------------------------
    def _kernel_aggregate(self, h: torch.Tensor, edge_index) -> torch.Tensor:
        """
        h : [N, d]           node embeddings
        returns z : [N, d]   kernel‑smoothed embeddings
        """
        a = F.normalize(self.aux, dim=0)                          # unit vector
        score = F.cosine_similarity(h, a.unsqueeze(0), dim=-1)    # s_i  (N)
        diff  = score.unsqueeze(0) - score.unsqueeze(1)           # (N,N)
        # using edge_index to sparsify the diff matrix
        # sparse_diff = torch.sparse_coo_tensor(indices=edge_index, values = diff, size=(diff.shape[0], diff.shape[1]), device=diff.device).coalesce()
        # sparse_diff = torch.zeros_like(diff)
        # mask[edge_index[0], edge_index[1]] = diff[edge_index[0], edge_index[1]]
        # sparse_diff = mask.to_sparse().coalesce()
        kappa = torch.exp(-(diff ** 2) / (self.sigma ** 2))       # eq.(3)
        z     = (kappa @ h) / kappa.sum(dim=1, keepdim=True)
        return z

    # ---------- forward pass -----------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h = self.conv2(h, edge_index)                              # h_i
        z = self._kernel_aggregate(h, edge_index)                              # z_i
        out = self.clsif(torch.cat([h, z], dim=-1))
        # out = h + z
        # out = F.relu(out)
        # out = F.dropout(out, p=self.dropout1, training=self.training)
        # out = self.clsif(out)
        return out