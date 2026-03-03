import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F


class BiasGATLayer(nn.Module):
    """Implementation of the bias GAT update mechanism."""
    def __init__(self, scalar_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.q_lin = nn.Linear(scalar_dim, scalar_dim)
        self.k_lin = nn.Linear(scalar_dim, scalar_dim)
        self.v_lin = nn.Linear(scalar_dim, scalar_dim)
        self.e_lin = nn.Linear(scalar_dim, scalar_dim)
        self.alpha_bn = nn.BatchNorm1d(scalar_dim)
        self.message_bn = nn.BatchNorm1d(scalar_dim)
        self.act = nn.Softplus()
        self.k_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.gate = nn.Sigmoid()

    def bias_gat_attn(
        self,
        src_feature: torch.Tensor,  # (num_edges, scalar_dim)
        dst_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
    ) -> torch.Tensor:
        # q, k, v: (num_edges, scalar_dim)
        q = self.q_lin(dst_feature)
        k = self.k_lin(src_feature)
        v = self.v_lin(src_feature)
        # edge_feature: (num_edges, scalar_dim)
        edge_feature_after_lin = self.e_lin(edge_feature)
        # attn: (num_edges, scalar_dim)
        attn = (
            torch.softmax(q * k, dim=-1)
            / torch.sqrt(torch.tensor(self.scalar_dim, dtype=q.dtype))
            + edge_feature_after_lin
        )
        # message: (num_edges, scalar_dim)
        message = attn * v
        return message, attn

    def bias_gat_update(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        src_feature = atom_feature[src]
        dst_feature = atom_feature[dst]
        message, attn = self.bias_gat_attn(src_feature, dst_feature, edge_feature)
        num_nodes = atom_feature.size(0)
        atom_feature = atom_feature + scatter(
            message, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        edge_feature = edge_feature + attn
        return atom_feature, edge_feature

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        return self.bias_gat_update(atom_feature, edge_feature, edge_index)


class ComformerLayer(nn.Module):
    """Implementation of the ComFormer update mechanism."""
    def __init__(self, scalar_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.q_lin = nn.Linear(scalar_dim, scalar_dim)
        self.k_lin = nn.Linear(scalar_dim, scalar_dim)
        self.v_lin = nn.Linear(scalar_dim, scalar_dim)
        self.e_lin = nn.Linear(scalar_dim, scalar_dim)
        self.alpha_bn = nn.BatchNorm1d(scalar_dim)
        self.message_bn = nn.BatchNorm1d(scalar_dim)
        self.act = nn.Softplus()
        self.k_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.gate = nn.Sigmoid()

    def comformer_node_attn(
        self,
        src_feature: torch.Tensor,  # (num_edges, scalar_dim)
        dst_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
    ) -> torch.Tensor:
        # _q: (num_edges, scalar_dim)
        _q = self.q_lin(dst_feature)
        # _k: (num_edges, scalar_dim * 3)
        _k = torch.stack(
            [
                self.k_lin(src_feature),
                self.k_lin(dst_feature),
                self.e_lin(edge_feature),
            ],
            dim=-1,
        ).view(-1, self.scalar_dim * 3)
        # _v: (num_edges, scalar_dim * 3)
        _v = torch.stack(
            [
                self.v_lin(src_feature),
                self.v_lin(dst_feature),
                self.e_lin(edge_feature),
            ],
            dim=-1,
        ).view(-1, self.scalar_dim * 3)
        # q, k, v: (num_edges, scalar_dim)
        q = _q
        k = self.k_mlp(_k)
        v = self.v_mlp(_v)
        # alpha: (num_edges, scalar_dim)
        scale = torch.tensor(self.scalar_dim, dtype=q.dtype)
        alpha = self.alpha_bn(q * k / torch.sqrt(scale))
        # message: (num_edges, scalar_dim)
        message = self.message_bn(self.gate(alpha) * v)
        return message

    def comformer_update(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ) -> torch.Tensor:
        src, dst = edge_index
        # src_feature: (num_edges, scalar_dim)
        src_feature = atom_feature[src]
        # dst_feature: (num_edges, scalar_dim)
        dst_feature = atom_feature[dst]
        # message: (num_edges, scalar_dim)
        message = self.comformer_node_attn(
            src_feature, dst_feature, edge_feature
        )
        # atom_feature: (num_nodes, scalar_dim)
        num_nodes = atom_feature.size(0)
        atom_feature = atom_feature + scatter(
            message, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        # edge_feature: (num_edges, scalar_dim)
        edge_feature = F.softplus(edge_feature + message)
        return atom_feature, edge_feature

    def forward(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ):
        return self.comformer_update(atom_feature, edge_feature, edge_index)


class InvariantLayer(nn.Module):
    """
    InvariantLayer class that creates either a BiasGATLayer or ComformerLayer
    based on the update_method parameter.
    """
    def __init__(self, update_method: str, scalar_dim: int):
        super().__init__()
        if update_method == 'bias_gat':
            self.layer = BiasGATLayer(scalar_dim)
        elif update_method == 'comformer':
            self.layer = ComformerLayer(scalar_dim)
        else:
            raise NotImplementedError(f'Not implemented yet: {update_method}')

    def forward(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ):
        return self.layer(atom_feature, edge_feature, edge_index)
