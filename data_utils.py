"""
Data loading, preprocessing, stratified splitting, CDF uncertainty prior computation,
sparse graph construction, truncated SVD (for LightGCL), and noise injection utilities
for authentic social recommendation benchmarks:
  - Last.fm (HetRec2011)
  - CiaoDVD (Guo et al., ASONAM 2014)
  - FilmTrust (Guo et al., IJCAI 2013)
"""

import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")


def _read_raw_dataset(name: str):
    name = name.lower()
    interactions = set()
    social_edges = set()

    if name == "lastfm":
        int_path = os.path.join(DATA_ROOT, "lastfm", "user_artists.dat")
        soc_path = os.path.join(DATA_ROOT, "lastfm", "user_friends.dat")
        with open(int_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    interactions.add((parts[0], parts[1]))
        with open(soc_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] != parts[1]:
                    social_edges.add((parts[0], parts[1]))

    elif name == "ciaodvd":
        int_path = os.path.join(DATA_ROOT, "ciaodvd", "movie-ratings.txt")
        soc_path = os.path.join(DATA_ROOT, "ciaodvd", "trusts.txt")
        with open(int_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    interactions.add((parts[0].strip(), parts[1].strip()))
        with open(soc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[0].strip() != parts[1].strip():
                    social_edges.add((parts[0].strip(), parts[1].strip()))

    elif name == "filmtrust":
        int_path = os.path.join(DATA_ROOT, "filmtrust", "ratings.txt")
        soc_path = os.path.join(DATA_ROOT, "filmtrust", "trust.txt")
        with open(int_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    interactions.add((parts[0], parts[1]))
        with open(soc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] != parts[1]:
                    social_edges.add((parts[0], parts[1]))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return list(interactions), list(social_edges)


def _k_core_filter(interactions, min_user_deg=3, min_item_deg=3):
    cur = interactions
    while True:
        u_cnt = defaultdict(int)
        i_cnt = defaultdict(int)
        for u, i in cur:
            u_cnt[u] += 1
            i_cnt[i] += 1
        nxt = [(u, i) for (u, i) in cur if u_cnt[u] >= min_user_deg and i_cnt[i] >= min_item_deg]
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur


class SocialRecDataset:
    def __init__(
        self,
        name: str,
        split_seed: int = 42,
        min_deg: int = 3,
        noise_ratio: float = 0.0,
        noise_seed: int = 2026,
        device: torch.device = torch.device("cuda"),
    ):
        self.name = name.lower()
        self.device = device
        self.noise_ratio = noise_ratio

        raw_ints, raw_soc = _read_raw_dataset(self.name)
        filtered_ints = _k_core_filter(raw_ints, min_user_deg=min_deg, min_item_deg=min_deg)

        # Contiguous ID mapping
        users = sorted({u for u, _ in filtered_ints})
        items = sorted({i for _, i in filtered_ints})
        self.u2id = {u: idx for idx, u in enumerate(users)}
        self.i2id = {i: idx for idx, i in enumerate(items)}
        self.num_users = len(users)
        self.num_items = len(items)

        user_all_items = defaultdict(list)
        for u, i in filtered_ints:
            user_all_items[self.u2id[u]].append(self.i2id[i])

        # Filter authentic social edges to active users (make undirected for symmetric influence)
        soc_set = set()
        for u, v in raw_soc:
            if u in self.u2id and v in self.u2id:
                uid, vid = self.u2id[u], self.u2id[v]
                if uid != vid:
                    soc_set.add((uid, vid))
                    soc_set.add((vid, uid))
        self.clean_social_edges = sorted(soc_set)

        # Stratified per-user 80% train / 10% val / 10% test split (fixed split_seed)
        rng = np.random.RandomState(split_seed)
        self.train_pairs = []
        self.train_user_items = defaultdict(list)
        self.val_user_items = {}
        self.test_user_items = {}

        for u in range(self.num_users):
            item_list = sorted(user_all_items[u])
            rng.shuffle(item_list)
            n = len(item_list)
            n_val = max(1, int(round(0.1 * n)))
            n_test = max(1, int(round(0.1 * n)))
            if n - n_val - n_test < 1:
                n_val = 1 if n >= 3 else 0
                n_test = 1 if n >= 2 else 0
            n_train = n - n_val - n_test

            tr = item_list[:n_train]
            va = item_list[n_train : n_train + n_val]
            te = item_list[n_train + n_val :]

            self.train_user_items[u] = tr
            for it in tr:
                self.train_pairs.append((u, it))
            if va:
                self.val_user_items[u] = set(va)
            if te:
                self.test_user_items[u] = set(te)

        self.train_user_sets = {u: set(its) for u, its in self.train_user_items.items()}

        # Optional noise injection into training interactions AND social edges (RQ3)
        self.noisy_train_pairs = list(self.train_pairs)
        self.noisy_social_edges = list(self.clean_social_edges)
        if noise_ratio > 0.0:
            nrng = np.random.RandomState(noise_seed)
            n_noise_int = int(len(self.train_pairs) * noise_ratio)
            added = 0
            while added < n_noise_int:
                u = int(nrng.randint(0, self.num_users))
                it = int(nrng.randint(0, self.num_items))
                if it not in self.train_user_sets[u] and it not in self.test_user_items.get(u, set()):
                    self.noisy_train_pairs.append((u, it))
                    added += 1

            n_noise_soc = int(len(self.clean_social_edges) * noise_ratio)
            soc_cur = set(self.clean_social_edges)
            added_s = 0
            while added_s < n_noise_soc:
                u = int(nrng.randint(0, self.num_users))
                v = int(nrng.randint(0, self.num_users))
                if u != v and (u, v) not in soc_cur:
                    soc_cur.add((u, v))
                    self.noisy_social_edges.append((u, v))
                    added_s += 1

        self.train_users_np = np.array([u for u, _ in self.noisy_train_pairs], dtype=np.int64)
        self.train_items_np = np.array([i for _, i in self.noisy_train_pairs], dtype=np.int64)

        # Build sparse graphs on device
        self.norm_adj = self._build_norm_bipartite_adj(self.noisy_train_pairs, device)
        self.social_adj, self.social_edges_tensor = self._build_social_graph(self.noisy_social_edges, device)

        # Compute Empirical CDF (ECDF) uncertainty priors strictly from training graph
        self.user_cdf_prior, self.item_cdf_prior = self._compute_ecdf_priors(device)

        # Compute truncated SVD (rank q=5) for LightGCL
        self.svd_u, self.svd_s, self.svd_v = self._compute_truncated_svd(rank=5, device=device)

    def _build_norm_bipartite_adj(self, pairs, device):
        U, I = self.num_users, self.num_items
        N = U + I
        u_idx = np.array([u for u, _ in pairs], dtype=np.int64)
        i_idx = np.array([i for _, i in pairs], dtype=np.int64) + U

        rows = np.concatenate([u_idx, i_idx])
        cols = np.concatenate([i_idx, u_idx])
        vals = np.ones(len(rows), dtype=np.float32)

        mat = sp.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
        deg = np.array(mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
        d_inv_sqrt[deg == 0] = 0.0
        D_inv = sp.diags(d_inv_sqrt)
        norm_mat = (D_inv @ mat @ D_inv).tocoo()

        indices = torch.tensor(
            np.vstack([norm_mat.row, norm_mat.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(norm_mat.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()

    def _build_social_graph(self, soc_edges, device):
        U = self.num_users
        if len(soc_edges) == 0:
            idx = torch.arange(U, dtype=torch.long, device=device)
            adj = torch.sparse_coo_tensor(
                torch.stack([idx, idx], dim=0),
                torch.ones(U, dtype=torch.float32, device=device),
                (U, U),
                device=device,
            ).coalesce()
            return adj, torch.zeros((2, 0), dtype=torch.long, device=device)

        rows = np.array([u for u, _ in soc_edges], dtype=np.int64)
        cols = np.array([v for _, v in soc_edges], dtype=np.int64)
        vals = np.ones(len(rows), dtype=np.float32)
        mat = sp.coo_matrix((vals, (rows, cols)), shape=(U, U), dtype=np.float32)
        # Add self-loop so isolated users retain their own representation
        mat = (mat + sp.eye(U, dtype=np.float32)).tocoo()
        deg = np.array(mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
        D_inv = sp.diags(d_inv_sqrt)
        norm_mat = (D_inv @ mat @ D_inv).tocoo()

        indices = torch.tensor(
            np.vstack([norm_mat.row, norm_mat.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(norm_mat.data, dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(indices, values, (U, U), device=device).coalesce()

        # Include self-loops in social_edges_tensor for GMM-weighted social perception graph
        self_u = np.arange(U, dtype=np.int64)
        all_r = np.concatenate([rows, self_u])
        all_c = np.concatenate([cols, self_u])
        edge_tensor = torch.tensor(np.vstack([all_r, all_c]), dtype=torch.long, device=device)
        return adj, edge_tensor

    def _compute_ecdf_priors(self, device, sigma_min=0.1, sigma_max=0.6):
        """
        Empirical CDF (ECDF) mapping of training degrees to bounded [0, 1] quantiles:
        q_v = F_hat(deg_v) => sigma_prior(v) = sigma_min + (sigma_max - sigma_min) * (1 - q_v).
        Sparse/cold-start nodes receive larger initial uncertainty; active nodes receive tighter priors.
        """
        u_int_deg = np.zeros(self.num_users, dtype=np.float64)
        i_int_deg = np.zeros(self.num_items, dtype=np.float64)
        for u, i in self.noisy_train_pairs:
            u_int_deg[u] += 1.0
            i_int_deg[i] += 1.0

        u_soc_deg = np.zeros(self.num_users, dtype=np.float64)
        for u, _ in self.noisy_social_edges:
            u_soc_deg[u] += 1.0

        # Combined user activity score (interaction + 0.5 * social)
        u_deg = u_int_deg + 0.5 * u_soc_deg

        def ecdf(arr):
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64) / len(arr)
            return ranks

        q_u = ecdf(u_deg)
        q_i = ecdf(i_int_deg)

        sig_u = sigma_min + (sigma_max - sigma_min) * (1.0 - q_u)
        sig_i = sigma_min + (sigma_max - sigma_min) * (1.0 - q_i)

        return (
            torch.tensor(sig_u, dtype=torch.float32, device=device),
            torch.tensor(sig_i, dtype=torch.float32, device=device),
        )

    def _compute_truncated_svd(self, rank=5, device=torch.device("cuda")):
        rows = self.train_users_np
        cols = self.train_items_np
        vals = np.ones(len(rows), dtype=np.float32)
        R = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_items), dtype=np.float32)
        u_deg = np.maximum(np.array(R.sum(axis=1)).flatten(), 1e-12)
        i_deg = np.maximum(np.array(R.sum(axis=0)).flatten(), 1e-12)
        R_norm = sp.diags(np.power(u_deg, -0.5)) @ R @ sp.diags(np.power(i_deg, -0.5))
        k = min(rank, min(self.num_users, self.num_items) - 2)
        u, s, vt = svds(R_norm.tocsc(), k=k)
        # Sort descending
        idx = np.argsort(-s)
        u = u[:, idx].copy()
        s = s[idx].copy()
        v = vt[idx, :].T.copy()
        return (
            torch.tensor(u, dtype=torch.float32, device=device),
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(v, dtype=torch.float32, device=device),
        )

    def get_sparsity_groups(self):
        """
        Stratifies test users into 4 groups by training interaction count.
        Uses clean, interpretable degree buckets.
        """
        degs = {u: len(self.train_user_items[u]) for u in self.test_user_items.keys()}
        vals = np.array(list(degs.values()))
        # Use standard thresholds (<10, 10-20, 20-50, >50) if well-populated, else quartiles
        g1 = [u for u, d in degs.items() if d < 10]
        g2 = [u for u, d in degs.items() if 10 <= d < 20]
        g3 = [u for u, d in degs.items() if 20 <= d <= 50]
        g4 = [u for u, d in degs.items() if d > 50]
        if min(len(g1), len(g2), len(g3), len(g4)) >= 15:
            return {"<10": g1, "10-20": g2, "20-50": g3, ">50": g4}
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        q1, q2, q3 = int(max(1, round(q25))), int(max(2, round(q50))), int(max(3, round(q75)))
        return {
            f"Q1 (<={q1})": [u for u, d in degs.items() if d <= q1],
            f"Q2 ({q1+1}-{q2})": [u for u, d in degs.items() if q1 < d <= q2],
            f"Q3 ({q2+1}-{q3})": [u for u, d in degs.items() if q2 < d <= q3],
            f"Q4 (>{q3})": [u for u, d in degs.items() if d > q3],
        }

    def stats(self):
        total_int = len(self.train_pairs) + sum(len(v) for v in self.val_user_items.values()) + sum(
            len(v) for v in self.test_user_items.values()
        )
        n_soc = len(self.clean_social_edges)
        int_density = 100.0 * total_int / (self.num_users * self.num_items)
        soc_density = 100.0 * n_soc / (self.num_users * max(1, self.num_users - 1))
        return {
            "dataset": self.name,
            "users": self.num_users,
            "items": self.num_items,
            "interactions": total_int,
            "train_interactions": len(self.train_pairs),
            "social_edges": n_soc,
            "interaction_density_pct": round(int_density, 4),
            "social_density_pct": round(soc_density, 4),
        }

    def sample_batch_negatives(self, batch_users_np: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Fast 1:1 uniform negative sampling excluding training positives."""
        neg = rng.randint(0, self.num_items, size=len(batch_users_np))
        for idx, (u, ni) in enumerate(zip(batch_users_np, neg)):
            pos_set = self.train_user_sets[u]
            while ni in pos_set:
                ni = int(rng.randint(0, self.num_items))
            neg[idx] = ni
        return neg
