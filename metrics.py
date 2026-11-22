"""
Evaluation metrics and statistical significance tests for DistSGCL.
Implements full-ranking Precision@K, Recall@K, NDCG@K, and Shannon Entropy@K on GPU,
plus paired t-test, Wilcoxon signed-rank test, and 95% confidence intervals.
"""

import math
import numpy as np
import torch
from scipy import stats


@torch.no_grad()
def evaluate_full_ranking(
    model,
    train_user_items: dict,
    eval_user_items: dict,
    num_items: int,
    ks=(10, 20),
    batch_size: int = 512,
    device: torch.device = torch.device("cuda"),
):
    """
    Full-ranking evaluation over all candidate items (excluding training items).
    Returns a dict with Precision@K, Recall@K, NDCG@K, and ShannonEntropy@K.
    """
    model.eval()
    max_k = max(ks)
    eval_users = sorted(eval_user_items.keys())
    if len(eval_users) == 0:
        return {}

    # Precompute item representations / cache on GPU
    cache = model.get_eval_cache()

    prec_sums = {k: 0.0 for k in ks}
    rec_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    item_rec_counts = {k: np.zeros(num_items, dtype=np.int64) for k in ks}

    # Precompute discount factors for NDCG
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
    cum_discounts = np.cumsum(discounts)

    total_users = len(eval_users)

    for start in range(0, total_users, batch_size):
        batch_u = eval_users[start : start + batch_size]
        u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)

        # (B, num_items)
        scores = model.predict_all_items(u_tensor, cache)

        # Mask training items with -inf
        for idx, u in enumerate(batch_u):
            tr_items = train_user_items.get(u)
            if tr_items:
                scores[idx, tr_items] = -1e9

        _, topk_indices = torch.topk(scores, k=max_k, dim=1)
        topk_np = topk_indices.cpu().numpy()

        for idx, u in enumerate(batch_u):
            gt = eval_user_items[u]
            n_gt = len(gt)
            if n_gt == 0:
                continue
            preds = topk_np[idx]
            hits = np.array([1.0 if item in gt else 0.0 for item in preds], dtype=np.float64)

            for k in ks:
                hits_k = hits[:k]
                n_hit = hits_k.sum()
                prec_sums[k] += n_hit / k
                rec_sums[k] += n_hit / n_gt

                dcg = float((hits_k * discounts[:k]).sum())
                idcg = float(cum_discounts[min(n_gt, k) - 1])
                ndcg_sums[k] += (dcg / idcg) if idcg > 0 else 0.0

                # Accumulate recommended item frequencies for Shannon Entropy
                for item_id in preds[:k]:
                    item_rec_counts[k][item_id] += 1

    results = {}
    for k in ks:
        results[f"Precision@{k}"] = prec_sums[k] / total_users
        results[f"Recall@{k}"] = rec_sums[k] / total_users
        results[f"NDCG@{k}"] = ndcg_sums[k] / total_users

        # Shannon Entropy@K = -sum_i p(i) log2 p(i)
        counts = item_rec_counts[k]
        total_recs = counts.sum()
        if total_recs > 0:
            probs = counts[counts > 0] / float(total_recs)
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0
        results[f"Entropy@{k}"] = entropy

    return results


@torch.no_grad()
def evaluate_by_user_groups(
    model,
    train_user_items: dict,
    eval_user_items: dict,
    user_groups: dict,
    num_items: int,
    k: int = 10,
    batch_size: int = 512,
    device: torch.device = torch.device("cuda"),
):
    """
    Evaluate Precision@k, Recall@k, NDCG@k separately for each sparsity group.
    user_groups: dict[group_name -> set/list of user_ids]
    """
    group_results = {}
    for gname, gusers in user_groups.items():
        sub_eval = {u: eval_user_items[u] for u in gusers if u in eval_user_items}
        if len(sub_eval) == 0:
            group_results[gname] = {
                f"Precision@{k}": 0.0,
                f"Recall@{k}": 0.0,
                f"NDCG@{k}": 0.0,
                "num_users": 0,
            }
            continue
        res = evaluate_full_ranking(
            model,
            train_user_items,
            sub_eval,
            num_items,
            ks=(k,),
            batch_size=batch_size,
            device=device,
        )
        res["num_users"] = len(sub_eval)
        group_results[gname] = res
    return group_results


def summarize_seeds(values):
    """
    Given a list/array of metric values across N seeds, returns:
    mean, std, ci95_half_width.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci95 = t_crit * std / math.sqrt(n)
    else:
        ci95 = 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def significance_tests(ours_vals, baseline_vals):
    """
    Runs paired two-sided t-test and Wilcoxon signed-rank test across seeds.
    """
    a = np.asarray(ours_vals, dtype=np.float64)
    b = np.asarray(baseline_vals, dtype=np.float64)
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"t_stat": 0.0, "t_p": 1.0, "w_stat": 0.0, "w_p": 1.0}

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "w_stat": float(w_stat),
        "w_p": float(w_p),
    }
