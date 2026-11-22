import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rdir = os.path.join(BASE_DIR, "results")

with open(os.path.join(rdir, "dataset_stats.json")) as f:
    dstats = json.load(f)
print("=== DATASET STATS ===")
for k, v in dstats.items():
    print(v)

with open(os.path.join(rdir, "rq1_overall.json")) as f:
    rq1 = json.load(f)

for ds in ["lastfm", "ciaodvd", "filmtrust"]:
    print(f"\n=== RQ1: {ds.upper()} ===")
    for m, s in rq1["summary"][ds].items():
        p10, p10s = s["Precision@10"]["mean"], s["Precision@10"]["std"]
        r10, r10s = s["Recall@10"]["mean"], s["Recall@10"]["std"]
        n10, n10s = s["NDCG@10"]["mean"], s["NDCG@10"]["std"]
        p20 = s["Precision@20"]["mean"]
        r20 = s["Recall@20"]["mean"]
        n20 = s["NDCG@20"]["mean"]
        h10 = s["Entropy@10"]["mean"]
        print(
            f"  {m:10s} | P@10={p10:.4f}±{p10s:.4f} R@10={r10:.4f}±{r10s:.4f} N@10={n10:.4f}±{n10s:.4f} "
            f"| P@20={p20:.4f} R@20={r20:.4f} N@20={n20:.4f} | H@10={h10:.3f}"
        )
    print("  Significance vs baselines (NDCG@10 / Recall@10):")
    for base in ["SimGCL", "XSimGCL", "LightGCL", "SEPT", "HCCF", "SGL", "LightGCN"]:
        sn = rq1["significance"][ds][base]["NDCG@10"]
        sr = rq1["significance"][ds][base]["Recall@10"]
        print(
            f"    vs {base:8s} | N@10: t={sn['t_stat']:6.3f}, t_p={sn['t_p']:.4e}, w_p={sn['w_p']:.4e} "
            f"| R@10: t={sr['t_stat']:6.3f}, t_p={sr['t_p']:.4e}, w_p={sr['w_p']:.4e}"
        )

with open(os.path.join(rdir, "rq2_sparsity.json")) as f:
    rq2 = json.load(f)
for ds in ["lastfm", "ciaodvd", "filmtrust"]:
    print(f"\n=== RQ2 SPARSITY: {ds.upper()} (NDCG@10) ===")
    models = list(rq2[ds].keys())
    groups = list(rq2[ds][models[0]].keys())
    print("  Groups:", {g: rq2[ds][models[0]][g]["num_users"] for g in groups})
    for m in models:
        vals = [f"{g}: {rq2[ds][m][g]['NDCG@10']['mean']:.4f}" for g in groups]
        print(f"  {m:10s} | " + " | ".join(vals))

with open(os.path.join(rdir, "rq3_noise.json")) as f:
    rq3 = json.load(f)
for ds in ["lastfm", "ciaodvd"]:
    print(f"\n=== RQ3 NOISE: {ds.upper()} (NDCG@10) ===")
    for m in ["LightGCN", "SEPT", "SimGCL", "XSimGCL", "DistSGCL"]:
        vals = [f"{int(float(nl)*100)}%={rq3[ds][nl][m]['NDCG@10']['mean']:.4f}" for nl in ["0.0", "0.1", "0.2", "0.3"]]
        drop = (rq3[ds]["0.3"][m]["NDCG@10"]["mean"] - rq3[ds]["0.0"][m]["NDCG@10"]["mean"]) / rq3[ds]["0.0"][m]["NDCG@10"]["mean"] * 100
        print(f"  {m:10s} | " + " | ".join(vals) + f" | Drop={drop:+.2f}%")

with open(os.path.join(rdir, "rq4_ablation.json")) as f:
    rq4 = json.load(f)
for ds in ["lastfm", "ciaodvd", "filmtrust"]:
    print(f"\n=== RQ4 ABLATION: {ds.upper()} ===")
    for v, s in rq4["metrics"][ds].items():
        print(
            f"  {v:24s} | P@10={s['Precision@10']['mean']:.4f}±{s['Precision@10']['std']:.4f} "
            f"| R@10={s['Recall@10']['mean']:.4f}±{s['Recall@10']['std']:.4f} "
            f"| N@10={s['NDCG@10']['mean']:.4f}±{s['NDCG@10']['std']:.4f}"
        )

print("\n=== RQ4 COMPONENT COST (LASTFM) ===")
for v, p in rq4["component_cost_lastfm"].items():
    print(f"  {v:24s} | {p}")

with open(os.path.join(rdir, "rq6_efficiency.json")) as f:
    rq6 = json.load(f)
print("\n=== RQ6 EFFICIENCY (LASTFM) ===")
for m, p in rq6["lastfm"].items():
    print(f"  {m:10s} | {p}")
