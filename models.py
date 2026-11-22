"""
PyTorch GPU implementations of DistSGCL and 10 competitive baselines:
  1. MF-BPR (Rendle et al., UAI 2009)
  2. NCF (He et al., WWW 2017)
  3. DiffNet (Wu et al., SIGIR 2019) -- Social GNN
  4. LightGCN (He et al., SIGIR 2020) -- Graph CF
  5. SGL (Wu et al., SIGIR 2021) -- Edge-dropout GCL
  6. SEPT (Yu et al., KDD 2021) -- Socially-Aware Self-Supervised Tri-Training
  7. SimGCL (Yu et al., SIGIR 2022) -- Uniform Noise GCL
  8. HCCF (Xia et al., CIKM 2022) -- Hypergraph Contrastive CF
  9. LightGCL (Cai et al., ICLR 2023) -- SVD-based GCL
  10. XSimGCL (Yu et al., TKDE 2023) -- Cross-layer Simple GCL
  11. DistSGCL (Proposed) + 5 Ablation Variants:
      - DistSGCL (Full)
      - w/o Gaussian Pre-train
      - w/o GMM Social Graph
      - w/o Uncertainty Attn
      - w/o Reparam GCL
      - w/o Dist Scoring
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Cosine InfoNCE loss with in-batch negatives."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.T) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def sparse_edge_dropout(adj: torch.Tensor, keep_prob: float = 0.8) -> torch.Tensor:
    """Stochastic edge dropout on a coalesced PyTorch sparse COO tensor."""
    if keep_prob >= 1.0:
        return adj
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    mask = torch.rand(values.size(0), device=values.device) < keep_prob
    if mask.sum() == 0:
        return adj
    new_indices = indices[:, mask]
    new_values = values[mask] / keep_prob
    return torch.sparse_coo_tensor(new_indices, new_values, adj.size(), device=adj.device).coalesce()


class BaseRecModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, l2_reg: float = 1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.l2_reg = l2_reg

    @staticmethod
    def init_emb(emb: nn.Embedding):
        nn.init.normal_(emb.weight, std=0.1)

    def get_eval_cache(self):
        raise NotImplementedError

    def predict_all_items(self, user_ids: torch.Tensor, cache):
        u_emb, i_emb = cache
        return torch.matmul(u_emb[user_ids], i_emb.T)

    def compute_loss(self, users, pos_items, neg_items):
        raise NotImplementedError


# ============================================================================
# 1. MF-BPR (Rendle et al., UAI 2009)
# ============================================================================
class MFBPR(BaseRecModel):
    def __init__(self, num_users, num_items, emb_dim=64, l2_reg=1e-4):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def get_eval_cache(self):
        return self.user_emb.weight.detach(), self.item_emb.weight.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        p = self.item_emb(pos_items)
        n = self.item_emb(neg_items)
        pos_s = (u * p).sum(dim=-1)
        neg_s = (u * n).sum(dim=-1)
        loss = bpr_loss(pos_s, neg_s)
        reg = (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / users.size(0)
        return loss + self.l2_reg * reg


# ============================================================================
# 2. NCF (He et al., WWW 2017)
# ============================================================================
class NCF(BaseRecModel):
    def __init__(self, num_users, num_items, emb_dim=64, l2_reg=1e-4):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.u_gmf = nn.Embedding(num_users, emb_dim)
        self.i_gmf = nn.Embedding(num_items, emb_dim)
        self.u_mlp = nn.Embedding(num_users, emb_dim)
        self.i_mlp = nn.Embedding(num_items, emb_dim)
        for emb in [self.u_gmf, self.i_gmf, self.u_mlp, self.i_mlp]:
            nn.init.xavier_uniform_(emb.weight)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
        )
        self.out = nn.Linear(emb_dim + emb_dim // 2, 1)

    def _score_pairs(self, users, items):
        gmf = self.u_gmf(users) * self.i_gmf(items)
        mlp_in = torch.cat([self.u_mlp(users), self.i_mlp(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        return self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)

    def get_eval_cache(self):
        return (
            self.u_gmf.weight.detach(),
            self.i_gmf.weight.detach(),
            self.u_mlp.weight.detach(),
            self.i_mlp.weight.detach(),
        )

    def predict_all_items(self, user_ids, cache):
        u_gmf, i_gmf, u_mlp, i_mlp = cache
        B = user_ids.size(0)
        I = self.num_items
        # Chunk over items to keep VRAM low
        scores = torch.empty((B, I), device=user_ids.device)
        ug = u_gmf[user_ids]  # (B, d)
        um = u_mlp[user_ids]  # (B, d)
        chunk = 1024
        for s in range(0, I, chunk):
            e = min(s + chunk, I)
            C = e - s
            gmf = ug.unsqueeze(1) * i_gmf[s:e].unsqueeze(0)  # (B, C, d)
            mlp_in = torch.cat(
                [um.unsqueeze(1).expand(B, C, -1), i_mlp[s:e].unsqueeze(0).expand(B, C, -1)],
                dim=-1,
            )
            mlp_out = self.mlp(mlp_in)
            scores[:, s:e] = self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
        return scores

    def compute_loss(self, users, pos_items, neg_items):
        pos_s = self._score_pairs(users, pos_items)
        neg_s = self._score_pairs(users, neg_items)
        loss = bpr_loss(pos_s, neg_s)
        reg = (
            self.u_gmf(users).pow(2).sum()
            + self.i_gmf(pos_items).pow(2).sum()
            + self.i_gmf(neg_items).pow(2).sum()
            + self.u_mlp(users).pow(2).sum()
            + self.i_mlp(pos_items).pow(2).sum()
            + self.i_mlp(neg_items).pow(2).sum()
        ) / users.size(0)
        return loss + self.l2_reg * reg


# ============================================================================
# 3. DiffNet (Wu et al., SIGIR 2019) -- Social GNN
# ============================================================================
class DiffNet(BaseRecModel):
    def __init__(self, num_users, num_items, norm_adj, social_adj, emb_dim=64, n_layers=2, l2_reg=1e-4):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.social_adj = social_adj
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.fusion_w = nn.Linear(2 * emb_dim, emb_dim)

    def propagate(self):
        # Social influence diffusion over user-user graph
        u = self.user_emb.weight
        for _ in range(self.n_layers):
            u_soc = torch.sparse.mm(self.social_adj, u)
            u = 0.5 * (u + u_soc)
        # Bipartite interaction aggregation
        all_emb = torch.cat([u, self.item_emb.weight], dim=0)
        agg = torch.sparse.mm(self.norm_adj, all_emb)
        u_int, i_int = torch.split(agg, [self.num_users, self.num_items])
        u_final = u + u_int
        i_final = self.item_emb.weight + i_int
        return u_final, i_final

    def get_eval_cache(self):
        u_emb, i_emb = self.propagate()
        return u_emb.detach(), i_emb.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_all, i_all = self.propagate()
        u = u_all[users]
        p = i_all[pos_items]
        n = i_all[neg_items]
        loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return loss + self.l2_reg * reg


# ============================================================================
# 4. LightGCN (He et al., SIGIR 2020)
# ============================================================================
class LightGCN(BaseRecModel):
    def __init__(self, num_users, num_items, norm_adj, emb_dim=64, n_layers=3, l2_reg=1e-4):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self, adj=None):
        adj = self.norm_adj if adj is None else adj
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(final_emb, [self.num_users, self.num_items])

    def get_eval_cache(self):
        u_emb, i_emb = self.propagate()
        return u_emb.detach(), i_emb.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_all, i_all = self.propagate()
        u, p, n = u_all[users], i_all[pos_items], i_all[neg_items]
        loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return loss + self.l2_reg * reg


# ============================================================================
# 5. SGL (Wu et al., SIGIR 2021) -- Stochastic Edge Dropout GCL
# ============================================================================
class SGL(LightGCN):
    def __init__(self, num_users, num_items, norm_adj, emb_dim=64, n_layers=3,
                 l2_reg=1e-4, cl_rate=0.1, tau=0.2, drop_rate=0.2):
        super().__init__(num_users, num_items, norm_adj, emb_dim, n_layers, l2_reg)
        self.cl_rate = cl_rate
        self.tau = tau
        self.drop_rate = drop_rate
        self.aug_adj1 = None
        self.aug_adj2 = None

    def refresh_augmented_graphs(self):
        self.aug_adj1 = sparse_edge_dropout(self.norm_adj, keep_prob=1.0 - self.drop_rate)
        self.aug_adj2 = sparse_edge_dropout(self.norm_adj, keep_prob=1.0 - self.drop_rate)

    def compute_loss(self, users, pos_items, neg_items):
        u_all, i_all = self.propagate()
        u, p, n = u_all[users], i_all[pos_items], i_all[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)

        a1 = self.aug_adj1 if self.aug_adj1 is not None else self.norm_adj
        a2 = self.aug_adj2 if self.aug_adj2 is not None else self.norm_adj
        u1, i1 = self.propagate(a1)
        u2, i2 = self.propagate(a2)
        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = info_nce_loss(u1[uniq_u], u2[uniq_u], self.tau) + info_nce_loss(
            i1[uniq_i], i2[uniq_i], self.tau
        )
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 6. SEPT (Yu et al., KDD 2021) -- Socially-Aware Self-Supervised Tri-Training
# ============================================================================
class SEPT(BaseRecModel):
    def __init__(self, num_users, num_items, norm_adj, social_adj, emb_dim=64,
                 n_layers=3, l2_reg=1e-4, cl_rate=0.05, tau=0.2, drop_rate=0.2):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.social_adj = social_adj
        self.n_layers = n_layers
        self.cl_rate = cl_rate
        self.tau = tau
        self.drop_rate = drop_rate
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.aug_adj = None
        self.aug_soc = None

    def refresh_augmented_graphs(self):
        self.aug_adj = sparse_edge_dropout(self.norm_adj, keep_prob=1.0 - self.drop_rate)
        self.aug_soc = sparse_edge_dropout(self.social_adj, keep_prob=1.0 - self.drop_rate)

    def _propagate_int(self, adj):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(final_emb, [self.num_users, self.num_items])

    def _propagate_soc(self, soc_adj):
        u = self.user_emb.weight
        embs = [u]
        for _ in range(2):
            u = torch.sparse.mm(soc_adj, u)
            embs.append(u)
        return torch.stack(embs, dim=1).mean(dim=1)

    def get_eval_cache(self):
        u_int, i_int = self._propagate_int(self.norm_adj)
        u_soc = self._propagate_soc(self.social_adj)
        return (u_int + 0.2 * u_soc).detach(), i_int.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_int, i_int = self._propagate_int(self.norm_adj)
        u_soc = self._propagate_soc(self.social_adj)
        u_final = u_int + 0.2 * u_soc

        u, p, n = u_final[users], i_int[pos_items], i_int[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))

        # Tri-view socially-aware contrastive alignment (interaction view vs social view + augmented view)
        a_int = self.aug_adj if self.aug_adj is not None else self.norm_adj
        a_soc = self.aug_soc if self.aug_soc is not None else self.social_adj
        u_aug, i_aug = self._propagate_int(a_int)
        u_soc_aug = self._propagate_soc(a_soc)

        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = (
            info_nce_loss(u_int[uniq_u], u_soc_aug[uniq_u], self.tau)
            + info_nce_loss(u_int[uniq_u], u_aug[uniq_u], self.tau)
            + info_nce_loss(i_int[uniq_i], i_aug[uniq_i], self.tau)
        ) / 3.0

        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 7. SimGCL (Yu et al., SIGIR 2022) -- Uniform Embedding Noise GCL
# ============================================================================
class SimGCL(LightGCN):
    def __init__(self, num_users, num_items, norm_adj, emb_dim=64, n_layers=3,
                 l2_reg=1e-4, cl_rate=0.1, tau=0.2, eps=0.1):
        super().__init__(num_users, num_items, norm_adj, emb_dim, n_layers, l2_reg)
        self.cl_rate = cl_rate
        self.tau = tau
        self.eps = eps

    def propagate_perturbed(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = []
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            noise = torch.rand_like(all_emb)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * self.eps
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(final_emb, [self.num_users, self.num_items])

    def compute_loss(self, users, pos_items, neg_items):
        u_all, i_all = self.propagate()
        u, p, n = u_all[users], i_all[pos_items], i_all[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))

        u1, i1 = self.propagate_perturbed()
        u2, i2 = self.propagate_perturbed()
        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = info_nce_loss(u1[uniq_u], u2[uniq_u], self.tau) + info_nce_loss(
            i1[uniq_i], i2[uniq_i], self.tau
        )

        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 8. HCCF (Xia et al., CIKM 2022) -- Hypergraph Contrastive CF
# ============================================================================
class HCCF(BaseRecModel):
    def __init__(self, num_users, num_items, norm_adj, emb_dim=64, n_layers=2,
                 n_hyper=64, l2_reg=1e-4, cl_rate=0.05, tau=0.2):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.cl_rate = cl_rate
        self.tau = tau
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        # Learnable global hyperedge projection matrices
        self.u_hyper = nn.Parameter(torch.empty(emb_dim, n_hyper))
        self.i_hyper = nn.Parameter(torch.empty(emb_dim, n_hyper))
        nn.init.xavier_uniform_(self.u_hyper)
        nn.init.xavier_uniform_(self.i_hyper)

    def propagate(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        g_embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            g_embs.append(all_emb)
        g_final = torch.stack(g_embs, dim=1).mean(dim=1)
        u_g, i_g = torch.split(g_final, [self.num_users, self.num_items])

        # Hypergraph global dependency propagation: H * H^T * E
        Hu = F.softmax(torch.matmul(self.user_emb.weight, self.u_hyper), dim=-1)  # (U, K)
        Hi = F.softmax(torch.matmul(self.item_emb.weight, self.i_hyper), dim=-1)  # (I, K)
        u_h = torch.matmul(Hu, torch.matmul(Hu.T, u_g))
        i_h = torch.matmul(Hi, torch.matmul(Hi.T, i_g))
        return u_g, i_g, u_h, i_h

    def get_eval_cache(self):
        u_g, i_g, u_h, i_h = self.propagate()
        return (u_g + 0.1 * u_h).detach(), (i_g + 0.1 * i_h).detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_g, i_g, u_h, i_h = self.propagate()
        u_final = u_g + 0.1 * u_h
        i_final = i_g + 0.1 * i_h
        u, p, n = u_final[users], i_final[pos_items], i_final[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))

        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = info_nce_loss(u_g[uniq_u], u_h[uniq_u], self.tau) + info_nce_loss(
            i_g[uniq_i], i_h[uniq_i], self.tau
        )
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 9. LightGCL (Cai et al., ICLR 2023) -- SVD-based Graph Contrastive Learning
# ============================================================================
class LightGCL(BaseRecModel):
    def __init__(self, num_users, num_items, norm_adj, svd_u, svd_s, svd_v,
                 emb_dim=64, n_layers=2, l2_reg=1e-4, cl_rate=0.1, tau=0.2):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.svd_u = svd_u  # (U, q)
        self.svd_s = svd_s  # (q,)
        self.svd_v = svd_v  # (I, q)
        self.n_layers = n_layers
        self.cl_rate = cl_rate
        self.tau = tau
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self):
        # Local GCN view
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        local_embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            local_embs.append(all_emb)
        local_final = torch.stack(local_embs, dim=1).mean(dim=1)
        u_loc, i_loc = torch.split(local_final, [self.num_users, self.num_items])

        # Global SVD-reconstructed view: U * diag(S) * V^T * E_I and V * diag(S) * U^T * E_U
        u_cur = self.user_emb.weight
        i_cur = self.item_emb.weight
        u_svd_list, i_svd_list = [u_cur], [i_cur]
        for _ in range(self.n_layers):
            vt_i = torch.matmul(self.svd_v.T, i_cur) * self.svd_s.unsqueeze(1)
            u_next = torch.matmul(self.svd_u, vt_i)
            ut_u = torch.matmul(self.svd_u.T, u_cur) * self.svd_s.unsqueeze(1)
            i_next = torch.matmul(self.svd_v, ut_u)
            u_cur, i_cur = u_next, i_next
            u_svd_list.append(u_cur)
            i_svd_list.append(i_cur)
        u_svd = torch.stack(u_svd_list, dim=1).mean(dim=1)
        i_svd = torch.stack(i_svd_list, dim=1).mean(dim=1)
        return u_loc, i_loc, u_svd, i_svd

    def get_eval_cache(self):
        u_loc, i_loc, _, _ = self.propagate()
        return u_loc.detach(), i_loc.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_loc, i_loc, u_svd, i_svd = self.propagate()
        u, p, n = u_loc[users], i_loc[pos_items], i_loc[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))

        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = info_nce_loss(u_loc[uniq_u], u_svd[uniq_u], self.tau) + info_nce_loss(
            i_loc[uniq_i], i_svd[uniq_i], self.tau
        )
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 10. XSimGCL (Yu et al., TKDE 2023) -- Cross-layer Simple GCL
# ============================================================================
class XSimGCL(LightGCN):
    def __init__(self, num_users, num_items, norm_adj, emb_dim=64, n_layers=3,
                 l2_reg=1e-4, cl_rate=0.15, tau=0.2, eps=0.1, layer_cl=1):
        super().__init__(num_users, num_items, norm_adj, emb_dim, n_layers, l2_reg)
        self.cl_rate = cl_rate
        self.tau = tau
        self.eps = eps
        self.layer_cl = layer_cl

    def propagate_with_cl(self, perturbed=False):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        cl_emb = all_emb
        for l in range(1, self.n_layers + 1):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            if perturbed:
                noise = torch.rand_like(all_emb)
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * self.eps
            embs.append(all_emb)
            if l == self.layer_cl:
                cl_emb = all_emb
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        u_all, i_all = torch.split(final_emb, [self.num_users, self.num_items])
        u_cl, i_cl = torch.split(cl_emb, [self.num_users, self.num_items])
        return u_all, i_all, u_cl, i_cl

    def get_eval_cache(self):
        u_all, i_all, _, _ = self.propagate_with_cl(perturbed=False)
        return u_all.detach(), i_all.detach()

    def compute_loss(self, users, pos_items, neg_items):
        u_clean, i_clean, _, _ = self.propagate_with_cl(perturbed=False)
        u_all, i_all, u_cl, i_cl = self.propagate_with_cl(perturbed=True)
        u, p, n = u_clean[users], i_clean[pos_items], i_clean[neg_items]
        rec_loss = bpr_loss((u * p).sum(dim=-1), (u * n).sum(dim=-1))

        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        cl_loss = info_nce_loss(u_all[uniq_u], u_cl[uniq_u], self.tau) + info_nce_loss(
            i_all[uniq_i], i_cl[uniq_i], self.tau
        )
        reg = (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos_items).pow(2).sum()
            + self.item_emb(neg_items).pow(2).sum()
        ) / users.size(0)
        return rec_loss + self.cl_rate * cl_loss + self.l2_reg * reg


# ============================================================================
# 11. DistSGCL (Proposed Uncertainty-Aware Distributional Social GCL Framework)
# ============================================================================
class DistSGCL(BaseRecModel):
    """
    DistSGCL: Uncertainty-Aware Social Recommendation via CDF-Calibrated Gaussian
    Distributional Pre-training, GMM-Weighted Social Perception Graph,
    Uncertainty-Gated Multi-Head Attention, Uncertainty-Guided Reparameterized GCL,
    and End-to-End Distributional Wasserstein-BPR Scoring.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        social_edges: torch.Tensor,  # (2, E_s) long tensor
        user_cdf_prior: torch.Tensor,  # (U,) float tensor in [sigma_min, sigma_max]
        item_cdf_prior: torch.Tensor,  # (I,) float tensor in [sigma_min, sigma_max]
        emb_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        n_gmm: int = 8,
        l2_reg: float = 1e-4,
        lambda1: float = 0.05,  # Distributional consistency (2-Wasserstein prior anchor)
        lambda2: float = 0.15,  # Uncertainty-guided reparameterized GCL weight
        tau: float = 0.15,      # InfoNCE temperature
        gamma_w: float = 0.05,  # Distributional covariance penalty in scoring
        eps_cl: float = 0.15,   # Reparameterized perturbation scale
        # Ablation switches
        use_gaussian_pretrain: bool = True,
        use_gmm_social: bool = True,
        use_uncertainty_attn: bool = True,
        use_reparam_gcl: bool = True,
        use_dist_scoring: bool = True,
    ):
        super().__init__(num_users, num_items, emb_dim, l2_reg)
        self.norm_adj = norm_adj
        self.register_buffer("social_edges", social_edges)
        self.register_buffer("user_cdf_prior", user_cdf_prior)
        self.register_buffer("item_cdf_prior", item_cdf_prior)

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_gmm = n_gmm
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = tau
        self.gamma_w = gamma_w
        self.eps_cl = eps_cl

        # Ablation flags
        self.use_gaussian_pretrain = use_gaussian_pretrain
        self.use_gmm_social = use_gmm_social
        self.use_uncertainty_attn = use_uncertainty_attn
        self.use_reparam_gcl = use_reparam_gcl
        self.use_dist_scoring = use_dist_scoring

        # Diagonal Gaussian parameters: mu and raw rho (sigma = softplus(rho) + 1e-4)
        self.user_mu = nn.Embedding(num_users, emb_dim)
        self.item_mu = nn.Embedding(num_items, emb_dim)
        self.user_rho = nn.Embedding(num_users, emb_dim)
        self.item_rho = nn.Embedding(num_items, emb_dim)

        nn.init.normal_(self.user_mu.weight, std=0.1)
        nn.init.normal_(self.item_mu.weight, std=0.1)

        # Initialize rho from CDF uncertainty prior (inverse softplus)
        with torch.no_grad():
            u_init = torch.log(torch.expm1(user_cdf_prior.clamp(min=1e-3))).unsqueeze(1).expand(-1, emb_dim)
            i_init = torch.log(torch.expm1(item_cdf_prior.clamp(min=1e-3))).unsqueeze(1).expand(-1, emb_dim)
            self.user_rho.weight.copy_(u_init.cpu())
            self.item_rho.weight.copy_(i_init.cpu())

        # Pre-trained distributional anchors (filled after Stage 1 pre-training)
        dev = social_edges.device
        self.register_buffer("anchor_u_mu", self.user_mu.weight.detach().clone())
        self.register_buffer("anchor_u_sigma", self._get_sigma(self.user_rho.weight).detach().clone())
        self.register_buffer("anchor_i_mu", self.item_mu.weight.detach().clone())
        self.register_buffer("anchor_i_sigma", self._get_sigma(self.item_rho.weight).detach().clone())

        # GMM prototype embeddings & user soft assignments (filled after Stage 1 GMM)
        self.register_buffer("gmm_proto_emb", torch.zeros(num_users, emb_dim, device=dev))
        self.register_buffer("gmm_edge_weights", torch.ones(social_edges.size(1), device=dev))
        self.social_adj = self._build_weighted_social_adj(self.gmm_edge_weights)

        # Track which users have real non-self social neighbors (avoid distorting isolated users)
        has_soc = torch.zeros(num_users, dtype=torch.float32, device=dev)
        if social_edges.size(1) > 0:
            non_self = social_edges[0] != social_edges[1]
            if non_self.any():
                has_soc.index_fill_(0, social_edges[0][non_self], 1.0)
        self.register_buffer("has_social_mask", has_soc.unsqueeze(1))  # (U, 1)

        # Stage 1 pre-training bias for Distributional BCE reconstruction
        self.pretrain_bias = nn.Parameter(torch.tensor(0.0))

        # Uncertainty gate slope eta > 0
        self.unc_eta = nn.Parameter(torch.tensor(0.5))

        # Stage 2 Lightweight Uncertainty-Aware Multi-Head Attention Fusion + Learnable GMM Prototypes
        self.head_dim = emb_dim // n_heads
        self.attn_query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.05)
        self.attn_unc_beta = nn.Parameter(torch.tensor(1.0))
        self.soc_mix_alpha = nn.Parameter(torch.tensor(-1.5))
        self.gmm_prototypes = nn.Parameter(torch.randn(n_gmm, emb_dim) * 0.1)

    @staticmethod
    def _get_sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-4

    def _build_weighted_social_adj(self, edge_weights: torch.Tensor) -> torch.Tensor:
        device = edge_weights.device
        U = self.num_users
        if self.social_edges.size(1) == 0:
            idx = torch.arange(U, device=device)
            return torch.sparse_coo_tensor(
                torch.stack([idx, idx], dim=0),
                torch.ones(U, device=device),
                (U, U),
                device=device,
            ).coalesce()

        src, dst = self.social_edges[0], self.social_edges[1]
        # Row-normalize weighted social edges + self-loop for numerical stability
        deg = torch.zeros(U, device=device)
        deg.index_add_(0, src, edge_weights)
        norm_w = edge_weights / (deg[src] + 1e-8)
        return torch.sparse_coo_tensor(
            self.social_edges, norm_w, (U, U), device=device
        ).coalesce()

    # ------------------------------------------------------------------
    # Stage 1: CDF-Calibrated Gaussian Distributional Pre-training + GMM
    # ------------------------------------------------------------------
    def pretrain_step(self, users, pos_items, neg_items):
        """
        One mini-batch step of Stage 1 (Graph-Aware Gaussian Distributional Pre-training):
        Uses 1-hop graph propagation + Distributional BPR + in-batch uniformity + KL prior
        alignment to calibrate (mu^{(0)}, sigma^{(0)}) without injecting 0-hop popularity bias.
        """
        all_mu = torch.cat([self.user_mu.weight, self.item_mu.weight], dim=0)
        prop_mu = torch.sparse.mm(self.norm_adj, all_mu)
        u_mu_all, i_mu_all = torch.split(0.5 * all_mu + 0.5 * prop_mu, [self.num_users, self.num_items])

        u_mu = u_mu_all[users]
        p_mu = i_mu_all[pos_items]
        n_mu = i_mu_all[neg_items]
        u_sig = self._get_sigma(self.user_rho(users))
        p_sig = self._get_sigma(self.item_rho(pos_items))
        n_sig = self._get_sigma(self.item_rho(neg_items))

        # Distributional BPR reconstruction aligned with Stage 4
        pos_s = self._dist_score_pairs(u_mu, u_sig, p_mu, p_sig)
        neg_s = self._dist_score_pairs(u_mu, u_sig, n_mu, n_sig)
        l_rec_pre = bpr_loss(pos_s, neg_s)

        # KL divergence to ECDF-calibrated prior N(0, (sigma_prior)^2 I)
        u_prior = self.user_cdf_prior[users].unsqueeze(1)
        i_prior = self.item_cdf_prior[pos_items].unsqueeze(1)
        kl_u = 0.5 * (
            (u_sig.pow(2) + 0.1 * u_mu.pow(2)) / (u_prior.pow(2) + 1e-6)
            - 1.0
            + 2.0 * torch.log((u_prior + 1e-6) / u_sig)
        ).mean()
        kl_i = 0.5 * (
            (p_sig.pow(2) + 0.1 * p_mu.pow(2)) / (i_prior.pow(2) + 1e-6)
            - 1.0
            + 2.0 * torch.log((i_prior + 1e-6) / p_sig)
        ).mean()

        # Mild hyperspherical uniformity during pre-training to prevent popularity collapse
        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)
        u_n = F.normalize(u_mu_all[uniq_u], dim=-1)
        i_n = F.normalize(i_mu_all[uniq_i], dim=-1)
        unif = (
            torch.pdist(u_n[:256], p=2).pow(2).mul(-2.0).exp().mean().log()
            + torch.pdist(i_n[:256], p=2).pow(2).mul(-2.0).exp().mean().log()
        ) * 0.02

        return l_rec_pre + 0.05 * (kl_u + kl_i) + unif

    @torch.no_grad()
    def finalize_pretraining_and_fit_gmm(self, seed: int = 42):
        """
        Rescales pre-trained means to well-conditioned manifold norm, snapshots Gaussian
        anchors, and fits K-component GMM over 1-hop graph-smoothed user embeddings to
        compute semantic social edge weights w^{GMM}_{u,v} and prototype representations.
        """
        device = self.user_mu.weight.device
        # Keep mean embeddings well-conditioned (std ~ 0.1) so Stage 2 GCN + GCL converge cleanly
        if self.use_gaussian_pretrain:
            u_std = self.user_mu.weight.std().clamp(min=1e-4)
            i_std = self.item_mu.weight.std().clamp(min=1e-4)
            self.user_mu.weight.mul_(0.1 / u_std)
            self.item_mu.weight.mul_(0.1 / i_std)

        self.anchor_u_mu.copy_(self.user_mu.weight.detach())
        self.anchor_u_sigma.copy_(self._get_sigma(self.user_rho.weight).detach())
        self.anchor_i_mu.copy_(self.item_mu.weight.detach())
        self.anchor_i_sigma.copy_(self._get_sigma(self.item_rho.weight).detach())

        if self.use_gmm_social and self.num_users >= self.n_gmm:
            # Fit GMM on 1-hop graph-smoothed user representations for richer collaborative semantics
            all_0 = torch.cat([self.user_mu.weight, self.item_mu.weight], dim=0)
            smooth_1 = torch.sparse.mm(self.norm_adj, all_0)[: self.num_users]
            u_np = (0.5 * self.user_mu.weight + 0.5 * smooth_1).detach().cpu().numpy()
            gmm = GaussianMixture(
                n_components=self.n_gmm,
                covariance_type="diag",
                max_iter=100,
                random_state=seed,
                reg_covar=1e-4,
            )
            gmm.fit(u_np)
            gamma = gmm.predict_proba(u_np)  # (U, K)
            means = gmm.means_  # (K, d)
            proto = gamma @ means  # (U, d)
            self.gmm_proto_emb.copy_(torch.tensor(proto, dtype=torch.float32, device=device))
            self.gmm_prototypes.data.copy_(torch.tensor(means, dtype=torch.float32, device=device))

            if self.social_edges.size(1) > 0:
                src = self.social_edges[0].cpu().numpy()
                dst = self.social_edges[1].cpu().numpy()
                # Semantic reliability weight w_{u,v}^{GMM} = gamma_u^T gamma_v + 0.1 floor
                w = np.sum(gamma[src] * gamma[dst], axis=1) + 0.1
                w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
                self.gmm_edge_weights.copy_(w_tensor)
                self.social_adj = self._build_weighted_social_adj(self.gmm_edge_weights)
        else:
            self.gmm_proto_emb.copy_(self.user_mu.weight.detach())
            self.gmm_edge_weights.fill_(1.0)
            self.social_adj = self._build_weighted_social_adj(self.gmm_edge_weights)

    # ------------------------------------------------------------------
    # Stage 2: Uncertainty-Gated Dual-View Propagation & Attention Fusion
    # ------------------------------------------------------------------
    def _propagate_interaction(self, u_mu0, i_mu0, u_sig0, i_sig0, perturbed: bool = False):
        all_mu = torch.cat([u_mu0, i_mu0], dim=0)
        all_sig = torch.cat([u_sig0, i_sig0], dim=0)

        # Centered relative uncertainty in [-0.4, 0.4] for adaptive anisotropic perturbation
        if perturbed:
            node_unc = all_sig.detach().mean(dim=-1, keepdim=True)
            rel_unc = 0.4 * torch.tanh(node_unc - node_unc.mean())
            sig_det = all_sig.detach()

        mu_list = [] if perturbed else [all_mu]
        cl_layer_mu = all_mu
        for l in range(1, self.n_layers + 1):
            all_mu = torch.sparse.mm(self.norm_adj, all_mu)
            if perturbed:
                eps = torch.rand_like(all_mu)
                # Uncertainty-guided anisotropic perturbation: modulates both direction & scale by learned sigma
                aniso = F.normalize(sig_det * eps, dim=-1)
                scale = self.eps_cl * (1.0 + rel_unc)
                all_mu = all_mu + torch.sign(all_mu) * aniso * scale
            mu_list.append(all_mu)
            if l == 1:
                cl_layer_mu = all_mu

        int_mu = torch.stack(mu_list, dim=1).mean(dim=1)
        u_mu_int, i_mu_int = torch.split(int_mu, [self.num_users, self.num_items])
        u_cl_l1, i_cl_l1 = torch.split(cl_layer_mu, [self.num_users, self.num_items])

        if perturbed:
            return u_mu_int, i_mu_int, u_cl_l1, i_cl_l1

        # 1-hop variance propagation (clean & fast: preserves calibrated diagonal uncertainty)
        all_var_1 = torch.sparse.mm(self.norm_adj, all_sig.pow(2)).clamp(min=1e-6)
        int_sig = torch.sqrt(0.5 * all_sig.pow(2) + 0.5 * all_var_1)
        u_sig_int, i_sig_int = torch.split(int_sig, [self.num_users, self.num_items])
        return u_mu_int, u_sig_int, i_mu_int, i_sig_int

    def propagate(self):
        u_mu0 = self.user_mu.weight
        i_mu0 = self.item_mu.weight
        u_sig0 = self._get_sigma(self.user_rho.weight)
        i_sig0 = self._get_sigma(self.item_rho.weight)

        u_mu_int, u_sig_int, i_mu_int, i_sig_int = self._propagate_interaction(
            u_mu0, i_mu0, u_sig0, i_sig0, perturbed=False
        )

        # Social view propagation over GMM-weighted social perception graph
        u_mu_soc = torch.sparse.mm(self.social_adj, u_mu_int)
        u_sig_soc = torch.sqrt(torch.sparse.mm(self.social_adj, u_sig_int.pow(2)).clamp(min=1e-6))

        # Dynamic differentiable GMM prototype context
        if self.use_gmm_social:
            # Soft assignment gamma_{u,c} via cosine similarity to learnable GMM prototypes
            sim_p = torch.matmul(F.normalize(u_mu_int, dim=-1), F.normalize(self.gmm_prototypes, dim=-1).T) / 0.2
            gamma_dyn = F.softmax(sim_p, dim=-1)  # (U, K)
            proto_ctx = torch.matmul(gamma_dyn, self.gmm_prototypes)  # (U, d)
        else:
            proto_ctx = u_mu_int

        # Uncertainty-Aware Multi-Head Attention Fusion:
        # Sparse/noisy users (high u_sig_int) borrow MORE from reliable social & GMM prototype views;
        # Dense/confident users (low u_sig_int) preserve their clean interaction embeddings!
        if self.use_uncertainty_attn:
            views = torch.stack([u_mu_int, u_mu_soc, proto_ctx], dim=1)  # (U, 3, d)
            U = self.num_users
            v_heads = views.view(U, 3, self.n_heads, self.head_dim)
            logits = (v_heads * self.attn_query.view(1, 1, self.n_heads, self.head_dim)).sum(dim=-1) / math.sqrt(self.head_dim)
            u_int_unc = u_sig_int.mean(dim=-1, keepdim=True)
            u_soc_unc = u_sig_soc.mean(dim=-1, keepdim=True)
            proto_unc = 0.5 * (u_int_unc + u_soc_unc)
            view_unc = torch.cat([u_int_unc, u_soc_unc, proto_unc], dim=1).unsqueeze(-1)
            logits = logits - F.softplus(self.attn_unc_beta) * view_unc
            attn_w = F.softmax(logits, dim=1).unsqueeze(-1)
            fused = (attn_w * v_heads).sum(dim=1).reshape(U, self.emb_dim)

            # Adaptive gate: higher when user interaction uncertainty exceeds social uncertainty
            unc_gate = torch.sigmoid(F.softplus(self.unc_eta) * (u_int_unc - u_soc_unc))
            mix = 0.20 * torch.sigmoid(self.soc_mix_alpha) * unc_gate * self.has_social_mask
            u_mu_final = (1.0 - mix) * u_mu_int + mix * fused
            u_sig_final = (1.0 - mix) * u_sig_int + mix * u_sig_soc
        else:
            mix = 0.08 * self.has_social_mask
            u_mu_final = (1.0 - mix) * u_mu_int + mix * u_mu_soc
            u_sig_final = (1.0 - mix) * u_sig_int + mix * u_sig_soc

        return u_mu_final, u_sig_final, i_mu_int, i_sig_int, u_mu_soc, proto_ctx

    # ------------------------------------------------------------------
    # Stage 4: Distributional Wasserstein-BPR Scoring (dimension-normalized)
    # ------------------------------------------------------------------
    def _dist_score_pairs(self, u_mu, u_sig, i_mu, i_sig):
        dot = (u_mu * i_mu).sum(dim=-1)
        if self.use_dist_scoring:
            cov_pen = (u_sig - i_sig).pow(2).mean(dim=-1)
            return dot - self.gamma_w * cov_pen
        return dot

    def get_eval_cache(self):
        u_mu, u_sig, i_mu, i_sig, _, _ = self.propagate()
        return u_mu.detach(), u_sig.detach(), i_mu.detach(), i_sig.detach()

    def predict_all_items(self, user_ids: torch.Tensor, cache):
        u_mu, u_sig, i_mu, i_sig = cache
        um = u_mu[user_ids]
        scores = torch.matmul(um, i_mu.T)
        if self.use_dist_scoring:
            us = u_sig[user_ids]
            u_sq = us.pow(2).sum(dim=-1, keepdim=True)
            i_sq = i_sig.pow(2).sum(dim=-1).unsqueeze(0)
            cov_dist = (u_sq + i_sq - 2.0 * torch.matmul(us, i_sig.T)).clamp(min=0.0) / float(self.emb_dim)
            scores = scores - self.gamma_w * cov_dist
        return scores

    # ------------------------------------------------------------------
    # Joint Loss: Distributional BPR + Consistency + Reparameterized GCL
    # ------------------------------------------------------------------
    def compute_loss(self, users, pos_items, neg_items):
        u_mu, u_sig, i_mu, i_sig, u_mu_soc, proto_ctx = self.propagate()

        # 1. Distributional Recommendation Loss (L_r)
        pos_s = self._dist_score_pairs(u_mu[users], u_sig[users], i_mu[pos_items], i_sig[pos_items])
        neg_s = self._dist_score_pairs(u_mu[users], u_sig[users], i_mu[neg_items], i_sig[neg_items])
        l_rec = bpr_loss(pos_s, neg_s)

        uniq_u = torch.unique(users)
        uniq_i = torch.unique(pos_items)

        # 2. Distributional Uncertainty Anchor Loss (L_dist)
        if self.use_gaussian_pretrain and self.lambda1 > 0:
            l_dist_u = (u_sig[uniq_u] - self.anchor_u_sigma[uniq_u]).pow(2).mean()
            l_dist_i = (i_sig[uniq_i] - self.anchor_i_sigma[uniq_i]).pow(2).mean()
            l_dist = 0.5 * (l_dist_u + l_dist_i)
        else:
            l_dist = torch.tensor(0.0, device=users.device)

        # 3. Layer-Wise Uncertainty-Guided Reparameterized Graph Contrastive Loss (L_InfoNCE)
        if self.use_reparam_gcl and self.lambda2 > 0:
            u_mu0, i_mu0 = self.user_mu.weight, self.item_mu.weight
            u_sig0, i_sig0 = self._get_sigma(self.user_rho.weight), self._get_sigma(self.item_rho.weight)
            z_u1, z_i1, z_u1_l1, z_i1_l1 = self._propagate_interaction(u_mu0, i_mu0, u_sig0, i_sig0, perturbed=True)
            z_u2, z_i2, _, _ = self._propagate_interaction(u_mu0, i_mu0, u_sig0, i_sig0, perturbed=True)

            # Dual-view anisotropic uniformity + cross-layer semantic alignment
            l_cl_u = 0.85 * info_nce_loss(z_u1[uniq_u], z_u2[uniq_u], self.tau) + 0.15 * info_nce_loss(
                z_u1[uniq_u], z_u1_l1[uniq_u], self.tau
            )
            l_cl_i = 0.85 * info_nce_loss(z_i1[uniq_i], z_i2[uniq_i], self.tau) + 0.15 * info_nce_loss(
                z_i1[uniq_i], z_i1_l1[uniq_i], self.tau
            )
            l_cl = l_cl_u + l_cl_i

            # True GMM Prototype-Level Contrastive Alignment (contrasts against K prototypes, zero false negatives)
            if self.use_gmm_social:
                u_norm = F.normalize(u_mu[uniq_u], dim=-1)
                p_norm = F.normalize(self.gmm_prototypes, dim=-1)
                logits_p = torch.matmul(u_norm, p_norm.T) / self.tau
                with torch.no_grad():
                    gamma_t = F.softmax(logits_p, dim=-1)
                l_proto = -(gamma_t * F.log_softmax(logits_p, dim=-1)).sum(dim=-1).mean()
                l_cl = l_cl + 0.05 * l_proto

            soc_u_mask = self.has_social_mask[uniq_u].squeeze(-1) > 0.5
            if soc_u_mask.sum() > 8:
                u_with_soc = uniq_u[soc_u_mask]
                l_cl = l_cl + 0.03 * info_nce_loss(u_mu[u_with_soc], u_mu_soc[u_with_soc], self.tau)
        else:
            l_cl = torch.tensor(0.0, device=users.device)

        # 4. L2 Regularization
        reg = (
            self.user_mu(users).pow(2).sum()
            + self.item_mu(pos_items).pow(2).sum()
            + self.item_mu(neg_items).pow(2).sum()
        ) / users.size(0)

        return l_rec + self.lambda1 * l_dist + self.lambda2 * l_cl + self.l2_reg * reg
