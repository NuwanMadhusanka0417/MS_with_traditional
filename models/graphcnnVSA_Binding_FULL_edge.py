import math
import sys

import torch
import torch.nn as nn
from torch.fft import fft, ifft

sys.path.append("models/")
from models.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers,
        delta,
        graph_pooling_type,
        neighbor_pooling_type,
        device,
        equation,
        edge_feat_dim=5,
        edge_projection_type="orthogonal",
        rng_seed=0,
    ):
        """
        GVFA GraphCNN with edge-conditioned message passing via concatenation.

        For each directed edge (src -> dst):
            message = W_msg @ concat(h[src], edge_H[e])
        then messages are aggregated at dst (sum/mean).

        edge_attr [E, edge_feat_dim] is projected to HV space with W_edge,
        then concatenated with the neighbor node HV (size 2D) and reduced
        back to D with a fixed random matrix W_msg.
        """
        super(GraphCNN, self).__init__()
        print("Input feature size: ", input_dim)
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = True
        self.delta = delta
        self.equation = equation
        self.edge_feat_dim = edge_feat_dim if edge_feat_dim else 0
        self.input_dim = input_dim

        if self.edge_feat_dim > 0:
            g = torch.Generator().manual_seed(rng_seed)

            # Project raw edge features [E, F_edge] -> edge HVs [E, D]
            if edge_projection_type == "orthogonal" and input_dim >= self.edge_feat_dim:
                A = torch.randn(input_dim, self.edge_feat_dim, generator=g)
                Q, _ = torch.linalg.qr(A)
                W_edge = Q[:, : self.edge_feat_dim].T  # (F_edge, D)
            else:
                W_edge = torch.randn(self.edge_feat_dim, input_dim, generator=g)
                W_edge = W_edge / math.sqrt(self.edge_feat_dim)
            self.register_buffer("W_edge", W_edge)

            # Reduce concat(neighbor_HV, edge_HV) [E, 2D] -> [E, D]
            W_msg = torch.randn(2 * input_dim, input_dim, generator=g)
            W_msg = W_msg / math.sqrt(2 * input_dim)
            self.register_buffer("W_msg", W_msg)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]])
        )
        return Adj_block.to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1.0 / len(graph.g)] * len(graph.g))
            else:
                elem.extend([1] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(
            idx, elem, torch.Size([len(batch_graph), start_idx[-1]])
        )
        return graph_pool.to(self.device)

    def __preprocess_edges(self, batch_graph):
        """Batched edge_index [2, E] and edge_attr [E, F_edge], with node offsets."""
        start_idx = [0]
        for i, g in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(g.g))
        ei_list, ea_list = [], []
        for i, g in enumerate(batch_graph):
            ei = getattr(g, "edge_index", None)
            ea = getattr(g, "edge_attr", None)
            if ei is None or ea is None or ei.numel() == 0 or ea.numel() == 0:
                continue
            off = start_idx[i]
            ei_list.append(ei.to(self.device) + off)
            ea_list.append(ea.to(self.device))
        if not ei_list:
            return None, None, start_idx
        return torch.cat(ei_list, dim=1), torch.cat(ea_list, dim=0), start_idx

    def _edge_message_pool(self, h_to_pool, edge_index, edge_H, num_nodes, average=False):
        """
        Concat-based edge-conditioned message passing.

        For each edge (src, dst):
            message = W_msg @ concat(h_to_pool[src], edge_H[e])
        then aggregate messages at dst (sum or mean).
        """
        E = edge_index.shape[1]
        D = h_to_pool.shape[1]
        src, dst = edge_index[0], edge_index[1]
        neighbor_h = h_to_pool[src]  # [E, D]

        # Concatenate neighbor node HV with that edge's HV → [E, 2D] → [E, D]
        messages = torch.cat([neighbor_h, edge_H], dim=1)
        messages = torch.mm(messages, self.W_msg)

        pooled = torch.zeros(num_nodes, D, device=h_to_pool.device, dtype=h_to_pool.dtype)
        pooled.index_add_(0, dst, messages)

        if average:
            degree = torch.zeros(num_nodes, 1, device=h_to_pool.device, dtype=h_to_pool.dtype)
            degree.index_add_(
                0,
                dst,
                torch.ones(E, 1, device=h_to_pool.device, dtype=h_to_pool.dtype),
            )
            pooled = pooled / degree.clamp(min=1.0)
        return pooled

    def maxpool(self, h, padded_neighbor_list):
        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def bind(self, x, y):
        fft_self = fft(x, dim=1)
        fft_other = fft(y, dim=1)
        product = torch.mul(fft_self, fft_other)
        result = ifft(product, dim=1)
        return torch.real(result)

    def _pool_neighbors(self, h_pool, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes):
        """Edge-concat pool when edges are available; else adjacency pool."""
        use_edges = (
            edge_index is not None
            and edge_H is not None
            and num_nodes is not None
            and hasattr(self, "W_msg")
        )
        avg = self.neighbor_pooling_type == "average"
        if use_edges:
            return self._edge_message_pool(h_pool, edge_index, edge_H, num_nodes, average=avg)
        if self.neighbor_pooling_type == "max":
            return self.maxpool(h_pool, padded_neighbor_list)
        pooled = torch.spmm(Adj_block, h_pool)
        if avg:
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        return pooled

    def next_layer_eps(
        self,
        h,
        layer,
        padded_neighbor_list=None,
        Adj_block=None,
        delta=1,
        equation=10,
        edge_index=None,
        edge_H=None,
        num_nodes=None,
    ):
        shift = 1

        if equation == 10:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled = self._pool_neighbors(
                rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes
            )
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h

        elif equation == 11:
            pooled = self._pool_neighbors(
                h, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes
            )
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h
            pooled = torch.roll(pooled, shifts=shift, dims=1)

        else:
            rotated = torch.roll(h.clone(), shifts=shift, dims=1)
            pooled = self._pool_neighbors(
                rotated, Adj_block, padded_neighbor_list, edge_index, edge_H, num_nodes
            )
            if delta == 1:
                pooled = self.bind(h, pooled) + h
            elif delta == 2:
                pooled = self.bind(h, pooled) + h + pooled
            else:
                pooled = pooled + h
            pooled = torch.roll(pooled, shifts=shift, dims=1)

        pooled = torch.sign(pooled)
        return pooled

    def forward(self, batch_graph, return_embedding=False):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        batched_ei, batched_ea, _ = self.__preprocess_edges(batch_graph)
        num_nodes = X_concat.shape[0]
        edge_index = None
        edge_H = None
        if (
            batched_ei is not None
            and batched_ea is not None
            and self.edge_feat_dim > 0
            and hasattr(self, "W_edge")
        ):
            edge_index = batched_ei
            # Project bond features to same HV dim as nodes
            edge_H = torch.mm(batched_ea.to(X_concat.dtype), self.W_edge)

        hidden_rep = [X_concat]
        h = X_concat
        for layer in range(self.num_layers - 1):
            h = self.next_layer_eps(
                h,
                layer,
                Adj_block=Adj_block,
                delta=self.delta,
                equation=self.equation,
                edge_index=edge_index,
                edge_H=edge_H,
                num_nodes=num_nodes,
            )
            hidden_rep.append(h)

        pooled_hS = []
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_hS.append(pooled_h)

        return torch.stack(pooled_hS, dim=0)
