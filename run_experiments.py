"""
Comprehensive GPU Experimental Suite for DistSGCL (Revised Kybernetika Submission).
Executes all 6 research questions:
  - RQ1: Overall Performance & Statistical Significance (11 models x 3 datasets x 5 seeds)
  - RQ2: Sparsity / Cold-Start Analysis (4 user sparsity groups across datasets)
  - RQ3: Noise Robustness Analysis (0%, 10%, 20%, 30% injected interaction & social noise)
  - RQ4: Comprehensive Ablation Study (5 ablation variants + Full DistSGCL x 3 datasets x 5 seeds)
  - RQ5: Hyperparameter Sensitivity Analysis (tau, (lambda1, lambda2) grid, GMM K)
  - RQ6: Computational Cost & Efficiency Profiling (Params, Epoch Time, Total Time, Inference Latency, Peak VRAM, FLOPs)

Saves all structured JSON results into the local results/ directory.
"""

import os
import sys
import json
import time
import copy
import random
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import SocialRecDataset
from metrics import (
    evaluate_full_ranking,
    evaluate_by_user_groups,
    summarize_seeds,
    significance_tests,
)
from models import (
    MFBPR,
    NCF,
    DiffNet,
    LightGCN,
    SGL,
    SEPT,
    SimGCL,
    HCCF,
    LightGCL,
    XSimGCL,
    DistSGCL,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEEDS = [42, 1024, 2024, 3407, 7777]
DATASETS = ["lastfm", "ciaodvd", "filmtrust"]

EMB_DIM = 64
N_LAYERS = 3
N_HEADS = 4
N_GMM = 8
BATCH_SIZE = 1024
LR = 1e-3
L2_REG = 1e-4
MAX_EPOCHS = 120
EVAL_EVERY = 2
PATIENCE = 10  # 10 validation checks (= 20 epochs without improvement)
PRETRAIN_EPOCHS = 3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_epoch_flops(model_name: str, ds: SocialRecDataset, emb_dim=64, n_layers=3, n_heads=4, n_gmm=8):
    """
    Analytical FLOPs estimate per training epoch (in MFLOPs) for forward propagation & scoring.
    """
    U, I = ds.num_users, ds.num_items
    E_int = len(ds.noisy_train_pairs) * 2  # undirected bipartite edges
    E_soc = len(ds.noisy_social_edges)
    B = BATCH_SIZE
    n_batches = max(1, (len(ds.noisy_train_pairs) + B - 1) // B)
    d = emb_dim

    # Standard BPR dot product per epoch: 2 * |R| * 2d
    bpr_flops = 4 * len(ds.noisy_train_pairs) * d
    # LightGCN sparse mm per batch: n_layers * 2 * E_int * d
    gcn_flops = n_batches * n_layers * 2 * E_int * d
    # Social GNN sparse mm per batch: 2 * 2 * E_soc * d
    soc_flops = n_batches * 2 * 2 * E_soc * d
    # InfoNCE in-batch (approx B_u ~ min(U, B)): 2 * B_u^2 * d
    bu = min(U, B)
    bi = min(I, B)
    cl_flops = n_batches * 2 * (bu * bu + bi * bi) * d

    if model_name == "MF-BPR":
        total = bpr_flops
    elif model_name == "NCF":
        total = len(ds.noisy_train_pairs) * 2 * (2 * d * d + d * (d // 2) + (d + d // 2))
    elif model_name == "DiffNet":
        total = soc_flops + n_batches * 2 * E_int * d + bpr_flops
    elif model_name == "LightGCN":
        total = gcn_flops + bpr_flops
    elif model_name == "SGL":
        total = 3 * gcn_flops + cl_flops + bpr_flops
    elif model_name == "SEPT":
        total = 2 * gcn_flops + 2 * soc_flops + 1.5 * cl_flops + bpr_flops
    elif model_name == "SimGCL":
        total = 3 * gcn_flops + cl_flops + bpr_flops
    elif model_name == "HCCF":
        h_flops = n_batches * 4 * (U + I) * 64 * d
        total = gcn_flops + h_flops + cl_flops + bpr_flops
    elif model_name == "LightGCL":
        svd_flops = n_batches * n_layers * 4 * (U + I) * 5 * d
        total = gcn_flops + svd_flops + cl_flops + bpr_flops
    elif model_name == "XSimGCL":
        total = gcn_flops + cl_flops + bpr_flops
    elif model_name.startswith("DistSGCL"):
        # Dual mean+var propagation (2x gcn + 2x soc) + 3-view MHA (U * 3 * 3 * d + U * 3 * 4 * d^2) + reparam GCL + dist BPR
        mha_flops = n_batches * U * (3 * 4 * d * d + 9 * d)
        total = 2 * gcn_flops + 2 * soc_flops + mha_flops + 1.5 * cl_flops + 2 * bpr_flops
    else:
        total = gcn_flops + bpr_flops

    return round(total / 1e6, 2)  # MFLOPs


def build_model(model_name: str, ds: SocialRecDataset, device: torch.device, overrides: dict = None):
    ov = overrides or {}
    U, I = ds.num_users, ds.num_items
    d = ov.get("emb_dim", EMB_DIM)
    L = ov.get("n_layers", N_LAYERS)

    if model_name == "MF-BPR":
        return MFBPR(U, I, emb_dim=d, l2_reg=L2_REG).to(device)
    elif model_name == "NCF":
        return NCF(U, I, emb_dim=d, l2_reg=L2_REG).to(device)
    elif model_name == "DiffNet":
        return DiffNet(U, I, ds.norm_adj, ds.social_adj, emb_dim=d, n_layers=2, l2_reg=L2_REG).to(device)
    elif model_name == "LightGCN":
        return LightGCN(U, I, ds.norm_adj, emb_dim=d, n_layers=L, l2_reg=L2_REG).to(device)
    elif model_name == "SGL":
        return SGL(U, I, ds.norm_adj, emb_dim=d, n_layers=L, l2_reg=L2_REG, cl_rate=0.02, tau=0.2, drop_rate=0.2).to(device)
    elif model_name == "SEPT":
        return SEPT(U, I, ds.norm_adj, ds.social_adj, emb_dim=d, n_layers=L, l2_reg=L2_REG, cl_rate=0.02, tau=0.2, drop_rate=0.2).to(device)
    elif model_name == "SimGCL":
        return SimGCL(U, I, ds.norm_adj, emb_dim=d, n_layers=L, l2_reg=L2_REG, cl_rate=0.02, tau=0.2, eps=0.1).to(device)
    elif model_name == "HCCF":
        return HCCF(U, I, ds.norm_adj, emb_dim=d, n_layers=2, n_hyper=64, l2_reg=L2_REG, cl_rate=0.02, tau=0.2).to(device)
    elif model_name == "LightGCL":
        return LightGCL(U, I, ds.norm_adj, ds.svd_u, ds.svd_s, ds.svd_v, emb_dim=d, n_layers=2, l2_reg=L2_REG, cl_rate=0.02, tau=0.2).to(device)
    elif model_name == "XSimGCL":
        return XSimGCL(U, I, ds.norm_adj, emb_dim=d, n_layers=L, l2_reg=L2_REG, cl_rate=0.02, tau=0.2, eps=0.1, layer_cl=1).to(device)
    elif model_name.startswith("DistSGCL"):
        kwargs = dict(
            num_users=U,
            num_items=I,
            norm_adj=ds.norm_adj,
            social_edges=ds.social_edges_tensor,
            user_cdf_prior=ds.user_cdf_prior,
            item_cdf_prior=ds.item_cdf_prior,
            emb_dim=d,
            n_layers=L,
            n_heads=ov.get("n_heads", N_HEADS),
            n_gmm=ov.get("n_gmm", N_GMM),
            l2_reg=L2_REG,
            lambda1=ov.get("lambda1", 0.02),
            lambda2=ov.get("lambda2", 0.02),
            tau=ov.get("tau", 0.2),
            gamma_w=ov.get("gamma_w", 0.05),
            eps_cl=ov.get("eps_cl", 0.1),
            use_gaussian_pretrain=ov.get("use_gaussian_pretrain", True),
            use_gmm_social=ov.get("use_gmm_social", True),
            use_uncertainty_attn=ov.get("use_uncertainty_attn", True),
            use_reparam_gcl=ov.get("use_reparam_gcl", True),
            use_dist_scoring=ov.get("use_dist_scoring", True),
        )
        return DistSGCL(**kwargs).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(
    model_name: str,
    ds: SocialRecDataset,
    seed: int,
    device: torch.device,
    overrides: dict = None,
    profile: bool = False,
):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    model = build_model(model_name, ds, device, overrides=overrides)

    if profile and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    n_train = len(ds.train_users_np)
    indices = np.arange(n_train)

    t_start = time.perf_counter()

    # Stage 1: Gaussian Distributional Pre-training + GMM (only for DistSGCL when enabled)
    pretrain_time = 0.0
    if isinstance(model, DistSGCL):
        if model.use_gaussian_pretrain:
            t_pre0 = time.perf_counter()
            pre_opt = torch.optim.Adam(
                [
                    {"params": [model.user_rho.weight, model.item_rho.weight, model.pretrain_bias], "lr": LR * 2.0},
                    {"params": [model.user_mu.weight, model.item_mu.weight], "lr": LR * 0.1},
                ]
            )
            model.train()
            for _ in range(PRETRAIN_EPOCHS):
                rng.shuffle(indices)
                for s in range(0, n_train, BATCH_SIZE):
                    b_idx = indices[s : s + BATCH_SIZE]
                    u_np = ds.train_users_np[b_idx]
                    p_np = ds.train_items_np[b_idx]
                    n_np = ds.sample_batch_negatives(u_np, rng)
                    u_t = torch.tensor(u_np, dtype=torch.long, device=device)
                    p_t = torch.tensor(p_np, dtype=torch.long, device=device)
                    n_t = torch.tensor(n_np, dtype=torch.long, device=device)
                    pre_opt.zero_grad()
                    loss = model.pretrain_step(u_t, p_t, n_t)
                    loss.backward()
                    pre_opt.step()
            model.finalize_pretraining_and_fit_gmm(seed=seed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            pretrain_time = time.perf_counter() - t_pre0
        else:
            # Standard Xavier init without pre-training (ablation)
            model.finalize_pretraining_and_fit_gmm(seed=seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_ndcg = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep0 = time.perf_counter()
        model.train()
        if hasattr(model, "refresh_augmented_graphs"):
            model.refresh_augmented_graphs()

        rng.shuffle(indices)
        for s in range(0, n_train, BATCH_SIZE):
            b_idx = indices[s : s + BATCH_SIZE]
            u_np = ds.train_users_np[b_idx]
            p_np = ds.train_items_np[b_idx]
            n_np = ds.sample_batch_negatives(u_np, rng)

            u_t = torch.tensor(u_np, dtype=torch.long, device=device)
            p_t = torch.tensor(p_np, dtype=torch.long, device=device)
            n_t = torch.tensor(n_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = model.compute_loss(u_t, p_t, n_t)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t_ep0)

        if epoch % EVAL_EVERY == 0:
            val_res = evaluate_full_ranking(
                model,
                ds.train_user_items,
                ds.val_user_items,
                ds.num_items,
                ks=(10,),
                device=device,
            )
            val_ndcg = val_res["NDCG@10"]
            if val_ndcg > best_val_ndcg + 1e-5:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

    total_train_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference latency profiling (full ranking over all test users)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf0 = time.perf_counter()
    test_res = evaluate_full_ranking(
        model,
        ds.train_user_items,
        ds.test_user_items,
        ds.num_items,
        ks=(10, 20),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_time_ms = (time.perf_counter() - t_inf0) * 1000.0

    peak_vram_mb = 0.0
    if profile and device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    prof_info = {
        "params": count_parameters(model),
        "pretrain_time_s": round(pretrain_time, 3),
        "avg_epoch_time_s": round(float(np.mean(epoch_times)), 4),
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_times),
        "total_train_time_s": round(total_train_time, 2),
        "inference_time_ms": round(inf_time_ms, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mflops_per_epoch": estimate_epoch_flops(model_name, ds),
    }

    return model, test_res, prof_info


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 72)
    print(f"DistSGCL Experimental Suite | Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # Load all 3 authentic datasets
    datasets = {}
    dataset_stats = {}
    for dname in DATASETS:
        ds = SocialRecDataset(dname, split_seed=42, min_deg=3, noise_ratio=0.0, device=device)
        datasets[dname] = ds
        dataset_stats[dname] = ds.stats()
        print(f"[Dataset] {ds.stats()}")

    with open(os.path.join(RESULTS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # =========================================================================
    # RQ1: Overall Performance + Statistical Significance (11 models x 3 datasets x 5 seeds)
    # Also collects RQ2 (Sparsity) and RQ6 (Efficiency Profiling on seed 42)
    # =========================================================================
    all_models = [
        "MF-BPR",
        "NCF",
        "DiffNet",
        "LightGCN",
        "SGL",
        "SEPT",
        "SimGCL",
        "HCCF",
        "LightGCL",
        "XSimGCL",
        "DistSGCL",
    ]
    sparsity_models = {"LightGCN", "SEPT", "SimGCL", "LightGCL", "XSimGCL", "DistSGCL"}

    rq1_raw = {d: {} for d in DATASETS}
    rq1_summary = {d: {} for d in DATASETS}
    rq1_sig = {d: {} for d in DATASETS}
    rq2_sparsity = {d: {} for d in DATASETS}
    rq6_efficiency = {d: {} for d in DATASETS}

    print("\n" + "=" * 72)
    print("RUNNING RQ1 (Overall Performance), RQ2 (Sparsity), RQ6 (Efficiency)")
    print("=" * 72)

    for dname in DATASETS:
        ds = datasets[dname]
        user_groups = ds.get_sparsity_groups()
        print(f"\n--- Dataset: {dname.upper()} ---")

        for mname in all_models:
            seed_metrics = []
            group_seed_metrics = []
            prof_seed0 = None

            for s_idx, seed in enumerate(SEEDS):
                model, res, prof = train_and_evaluate(
                    mname, ds, seed, device, profile=(s_idx == 0)
                )
                seed_metrics.append(res)
                if s_idx == 0:
                    prof_seed0 = prof
                if mname in sparsity_models:
                    gres = evaluate_by_user_groups(
                        model, ds.train_user_items, ds.test_user_items, user_groups, ds.num_items, k=10, device=device
                    )
                    group_seed_metrics.append(gres)

            rq1_raw[dname][mname] = seed_metrics
            rq6_efficiency[dname][mname] = prof_seed0

            # Summarize across seeds
            summary = {}
            for metric_key in seed_metrics[0].keys():
                vals = [m[metric_key] for m in seed_metrics]
                summary[metric_key] = summarize_seeds(vals)
            rq1_summary[dname][mname] = summary

            print(
                f"  {mname:10s} | P@10: {summary['Precision@10']['mean']:.4f}±{summary['Precision@10']['std']:.4f} "
                f"| R@10: {summary['Recall@10']['mean']:.4f}±{summary['Recall@10']['std']:.4f} "
                f"| N@10: {summary['NDCG@10']['mean']:.4f}±{summary['NDCG@10']['std']:.4f} "
                f"| H@10: {summary['Entropy@10']['mean']:.3f} "
                f"| Time: {prof_seed0['total_train_time_s']:.1f}s"
            )

            # Aggregate sparsity groups across 5 seeds
            if mname in sparsity_models:
                g_agg = {}
                for gname in user_groups.keys():
                    g_agg[gname] = {
                        "num_users": group_seed_metrics[0][gname]["num_users"],
                        "Precision@10": summarize_seeds([gm[gname]["Precision@10"] for gm in group_seed_metrics]),
                        "Recall@10": summarize_seeds([gm[gname]["Recall@10"] for gm in group_seed_metrics]),
                        "NDCG@10": summarize_seeds([gm[gname]["NDCG@10"] for gm in group_seed_metrics]),
                    }
                rq2_sparsity[dname][mname] = g_agg

        # Statistical significance tests: DistSGCL vs every baseline
        ours_seeds = rq1_raw[dname]["DistSGCL"]
        for mname in all_models:
            if mname == "DistSGCL":
                continue
            base_seeds = rq1_raw[dname][mname]
            rq1_sig[dname][mname] = {}
            for mk in ["Precision@10", "Recall@10", "NDCG@10", "Precision@20", "Recall@20", "NDCG@20", "Entropy@10"]:
                rq1_sig[dname][mname][mk] = significance_tests(
                    [m[mk] for m in ours_seeds],
                    [m[mk] for m in base_seeds],
                )

    with open(os.path.join(RESULTS_DIR, "rq1_overall.json"), "w") as f:
        json.dump({"raw": rq1_raw, "summary": rq1_summary, "significance": rq1_sig}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "rq2_sparsity.json"), "w") as f:
        json.dump(rq2_sparsity, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "rq6_efficiency.json"), "w") as f:
        json.dump(rq6_efficiency, f, indent=2)

    # =========================================================================
    # RQ3: Noise Robustness Analysis (0%, 10%, 20%, 30% noise on Last.fm & CiaoDVD)
    # =========================================================================
    print("\n" + "=" * 72)
    print("RUNNING RQ3 (Noise Robustness Analysis)")
    print("=" * 72)

    noise_levels = [0.0, 0.1, 0.2, 0.3]
    noise_models = ["LightGCN", "SEPT", "SimGCL", "XSimGCL", "DistSGCL"]
    noise_datasets = ["lastfm", "ciaodvd"]
    rq3_noise = {d: {str(nl): {} for nl in noise_levels} for d in noise_datasets}

    for dname in noise_datasets:
        print(f"\n--- Noise Robustness on {dname.upper()} ---")
        for nl in noise_levels:
            if nl == 0.0:
                # Reuse RQ1 results for 0% noise
                for mname in noise_models:
                    rq3_noise[dname]["0.0"][mname] = {
                        mk: rq1_summary[dname][mname][mk]
                        for mk in ["Precision@10", "Recall@10", "NDCG@10"]
                    }
                continue

            noisy_ds = SocialRecDataset(dname, split_seed=42, min_deg=3, noise_ratio=nl, noise_seed=2026, device=device)
            for mname in noise_models:
                s_res = []
                for seed in SEEDS[:3]:  # 3 seeds per noise level for fast, tight error bars
                    _, res, _ = train_and_evaluate(mname, noisy_ds, seed, device)
                    s_res.append(res)
                rq3_noise[dname][str(nl)][mname] = {
                    mk: summarize_seeds([r[mk] for r in s_res])
                    for mk in ["Precision@10", "Recall@10", "NDCG@10"]
                }
                print(
                    f"  Noise {int(nl*100):2d}% | {mname:9s} | N@10: {rq3_noise[dname][str(nl)][mname]['NDCG@10']['mean']:.4f}"
                )

    with open(os.path.join(RESULTS_DIR, "rq3_noise.json"), "w") as f:
        json.dump(rq3_noise, f, indent=2)

    # =========================================================================
    # RQ4: Comprehensive Ablation Study (5 variants + Full DistSGCL x 3 datasets x 5 seeds)
    # Also profiles component computational overhead!
    # =========================================================================
    print("\n" + "=" * 72)
    print("RUNNING RQ4 (Comprehensive Ablation Study + Component Cost Profiling)")
    print("=" * 72)

    ablation_configs = {
        "w/o Gaussian Pre-train": {"use_gaussian_pretrain": False},
        "w/o GMM Social Graph": {"use_gmm_social": False},
        "w/o Uncertainty Attn": {"use_uncertainty_attn": False},
        "w/o Reparam GCL": {"use_reparam_gcl": False},
        "w/o Dist Scoring": {"use_dist_scoring": False},
    }
    rq4_ablation = {d: {} for d in DATASETS}
    rq4_comp_cost = {}

    for dname in DATASETS:
        ds = datasets[dname]
        print(f"\n--- Ablation on {dname.upper()} ---")
        rq4_ablation[dname]["DistSGCL (Full)"] = {
            mk: rq1_summary[dname]["DistSGCL"][mk]
            for mk in ["Precision@10", "Recall@10", "NDCG@10", "Entropy@10"]
        }
        if dname == "lastfm":
            rq4_comp_cost["DistSGCL (Full)"] = rq6_efficiency["lastfm"]["DistSGCL"]

        for vname, ov in ablation_configs.items():
            s_res = []
            prof0 = None
            for s_idx, seed in enumerate(SEEDS):
                _, res, prof = train_and_evaluate(
                    "DistSGCL", ds, seed, device, overrides=ov, profile=(dname == "lastfm" and s_idx == 0)
                )
                s_res.append(res)
                if s_idx == 0:
                    prof0 = prof
            rq4_ablation[dname][vname] = {
                mk: summarize_seeds([r[mk] for r in s_res])
                for mk in ["Precision@10", "Recall@10", "NDCG@10", "Entropy@10"]
            }
            if dname == "lastfm":
                rq4_comp_cost[vname] = prof0
            print(
                f"  {vname:24s} | P@10: {rq4_ablation[dname][vname]['Precision@10']['mean']:.4f} "
                f"| R@10: {rq4_ablation[dname][vname]['Recall@10']['mean']:.4f} "
                f"| N@10: {rq4_ablation[dname][vname]['NDCG@10']['mean']:.4f}"
            )

    with open(os.path.join(RESULTS_DIR, "rq4_ablation.json"), "w") as f:
        json.dump({"metrics": rq4_ablation, "component_cost_lastfm": rq4_comp_cost}, f, indent=2)

    # =========================================================================
    # RQ5: Hyperparameter Sensitivity Analysis (Last.fm)
    # 1) Temperature tau in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    # 2) Loss weights (lambda1, lambda2) grid
    # 3) GMM components K in [4, 8, 16, 32]
    # =========================================================================
    print("\n" + "=" * 72)
    print("RUNNING RQ5 (Hyperparameter Sensitivity Analysis on Last.fm)")
    print("=" * 72)

    ds_lf = datasets["lastfm"]
    rq5_sens = {"tau": {}, "lambda_grid": {}, "gmm_k": {}}

    for tau_val in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
        s_res = []
        for seed in SEEDS[:3]:
            _, res, _ = train_and_evaluate("DistSGCL", ds_lf, seed, device, overrides={"tau": tau_val})
            s_res.append(res)
        rq5_sens["tau"][str(tau_val)] = {
            mk: summarize_seeds([r[mk] for r in s_res])
            for mk in ["Precision@10", "Recall@10", "NDCG@10"]
        }
        print(f"  tau={tau_val:4.2f} | N@10={rq5_sens['tau'][str(tau_val)]['NDCG@10']['mean']:.4f}")

    l1_vals = [0.005, 0.01, 0.02, 0.05, 0.1]
    l2_vals = [0.01, 0.02, 0.03, 0.05, 0.1]
    for l1 in l1_vals:
        for l2 in l2_vals:
            key = f"{l1}_{l2}"
            _, res, _ = train_and_evaluate("DistSGCL", ds_lf, 42, device, overrides={"lambda1": l1, "lambda2": l2})
            rq5_sens["lambda_grid"][key] = {
                "lambda1": l1,
                "lambda2": l2,
                "Precision@10": res["Precision@10"],
                "Recall@10": res["Recall@10"],
                "NDCG@10": res["NDCG@10"],
            }
            print(f"  l1={l1:4.2f}, l2={l2:4.2f} | N@10={res['NDCG@10']:.4f}")

    for k_val in [4, 8, 16, 32]:
        s_res = []
        for seed in SEEDS[:3]:
            _, res, _ = train_and_evaluate("DistSGCL", ds_lf, seed, device, overrides={"n_gmm": k_val})
            s_res.append(res)
        rq5_sens["gmm_k"][str(k_val)] = {
            mk: summarize_seeds([r[mk] for r in s_res])
            for mk in ["Precision@10", "Recall@10", "NDCG@10"]
        }
        print(f"  GMM K={k_val:2d} | N@10={rq5_sens['gmm_k'][str(k_val)]['NDCG@10']['mean']:.4f}")

    with open(os.path.join(RESULTS_DIR, "rq5_sensitivity.json"), "w") as f:
        json.dump(rq5_sens, f, indent=2)

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
