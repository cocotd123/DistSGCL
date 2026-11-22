"""
Standalone RQ5 (Hyperparameter Sensitivity Analysis on Last.fm) with incremental saving.
"""
import os
import sys
import json
import warnings
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import SocialRecDataset
from metrics import summarize_seeds
from run_experiments import train_and_evaluate, SEEDS, RESULTS_DIR


def save_rq5(rq5_sens):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "rq5_sensitivity.json"), "w") as f:
        json.dump(rq5_sens, f, indent=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_lf = SocialRecDataset("lastfm", split_seed=42, min_deg=3, noise_ratio=0.0, device=device)
    rq5_sens = {"tau": {}, "lambda_grid": {}, "gmm_k": {}}

    print("RUNNING RQ5: Temperature tau (3 seeds each)")
    for tau_val in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        s_res = []
        for seed in SEEDS[:3]:
            _, res, _ = train_and_evaluate("DistSGCL", ds_lf, seed, device, overrides={"tau": tau_val})
            s_res.append(res)
        rq5_sens["tau"][str(tau_val)] = {
            mk: summarize_seeds([r[mk] for r in s_res])
            for mk in ["Precision@10", "Recall@10", "NDCG@10"]
        }
        save_rq5(rq5_sens)
        print(f"  tau={tau_val:4.2f} | N@10={rq5_sens['tau'][str(tau_val)]['NDCG@10']['mean']:.4f}", flush=True)

    print("RUNNING RQ5: (lambda1, lambda2) 4x4 grid")
    l1_vals = [0.005, 0.02, 0.05, 0.1]
    l2_vals = [0.01, 0.02, 0.05, 0.1]
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
            save_rq5(rq5_sens)
            print(f"  l1={l1:5.3f}, l2={l2:4.2f} | N@10={res['NDCG@10']:.4f}", flush=True)

    print("RUNNING RQ5: GMM components K (3 seeds each)")
    for k_val in [4, 8, 16, 32]:
        s_res = []
        for seed in SEEDS[:3]:
            _, res, _ = train_and_evaluate("DistSGCL", ds_lf, seed, device, overrides={"n_gmm": k_val})
            s_res.append(res)
        rq5_sens["gmm_k"][str(k_val)] = {
            mk: summarize_seeds([r[mk] for r in s_res])
            for mk in ["Precision@10", "Recall@10", "NDCG@10"]
        }
        save_rq5(rq5_sens)
        print(f"  GMM K={k_val:2d} | N@10={rq5_sens['gmm_k'][str(k_val)]['NDCG@10']['mean']:.4f}", flush=True)

    print("RQ5 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
