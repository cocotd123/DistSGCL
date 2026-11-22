# DistSGCL

Official PyTorch implementation and benchmark datasets for:

> **DistSGCL: Uncertainty-Aware Social Recommendation via Gaussian Distributional Pre-training and Graph Contrastive Learning**  
> Mengjie Zhao and Ying Chen\*  
> *Kybernetika* (submitted)  
> \*Corresponding Author: Ying Chen (`superkindnice@outlook.com`), School of Data Science, Chengdu Technological University

## Overview

DistSGCL (*Distributional Social Graph Contrastive Learning*) models each user and item as a $d$-dimensional diagonal Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}_v, \mathrm{diag}(\boldsymbol{\sigma}_v^2))$ and couples uncertainty modeling across four stages:

1. **ECDF-Calibrated Distributional Pre-training & GMM Social Weighting**: Maps heavy-tailed interaction degree counts to bounded variance priors via an Empirical Cumulative Distribution Function (ECDF), runs 1-hop graph-aware KL-regularized pre-training, and fits a $K$-component diagonal Gaussian Mixture Model (GMM) to reweight explicit social edges by posterior semantic similarity.
2. **Multi-View Distributional Propagation & Inverse-Variance Attention**: Propagates means and variances across the bipartite interaction graph, the GMM-weighted social graph, and differentiable semantic prototypes, then fuses views through an $H$-head inverse-variance self-attention gate with a cold-start residual path.
3. **Uncertainty-Guided Anisotropic Graph Contrastive Learning**: Perturbs each GCN layer along node-specific high-variance coordinates ($\boldsymbol{\sigma}_v \odot \boldsymbol{\epsilon}_v^{(l)}$), paired with cross-layer contrast and zero-false-negative prototype alignment.
4. **End-to-End 2-Wasserstein Ranking Optimization**: Scores user–item pairs via a dimension-normalized 2-Wasserstein objective inside Bayesian Personalized Ranking (BPR) so gradients update both preference means and uncertainty scales.

## Repository Structure

```text
DistSGCL/
├── data/                         # Public benchmark datasets (~4.55 MB total)
│   ├── lastfm/                   # HetRec2011 Last.fm (user_artists.dat, user_friends.dat, readme.txt)
│   ├── ciaodvd/                  # CiaoDVD (movie-ratings.txt, trusts.txt, readme.txt)
│   └── filmtrust/                # FilmTrust (ratings.txt, trust.txt, readme.txt)
├── data_utils.py                 # Data loader, 3-core filtering, stratified splits, ECDF priors, SVD, noise injection
├── metrics.py                    # Full-ranking Precision@K, Recall@K, NDCG@K, Entropy@K, t-test & Wilcoxon tests
├── models.py                     # PyTorch implementations of DistSGCL (full + 5 ablations) and 10 baselines
├── run_experiments.py            # Main experimental suite for RQ1--RQ6 (5 seeds x 3 datasets)
├── run_rq5.py                    # Standalone runner for RQ5 hyperparameter sensitivity
├── run_additional_analyses.py    # Runner for RQ7 (uncertainty calibration), RQ8 (depth/width), RQ9 (convergence/multi-K)
├── summarize_results.py          # Prints summary tables from locally generated results/*.json
├── requirements.txt
└── LICENSE
```

## Datasets

All three public benchmarks are placed under `data/` together with their original `readme.txt` attribution files. Following standard collaborative filtering practice, `data_utils.py` applies 3-core filtering (`min_deg=3`), binary interaction conversion, and per-user stratified 80% / 10% / 10% train/validation/test splitting (`split_seed=42`).

| Dataset | Users | Items | Interactions (Train) | Social Edges | Interaction Density | Social Density | Source |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Last.fm** | 1,875 | 4,613 | 77,465 (61,867) | 25,182 | 0.8956% | 0.7167% | HetRec 2011 (Cantador et al., RecSys 2011) |
| **CiaoDVD** | 3,958 | 4,158 | 41,904 (30,292) | 14,886 | 0.2546% | 0.0950% | Guo et al., ASONAM 2014 |
| **FilmTrust** | 1,324 | 782 | 33,319 (26,365) | 1,994 | 3.2181% | 0.1138% | Guo et al., IJCAI 2013 |

Please cite the original dataset papers when using these benchmarks (see each subdirectory's `readme.txt`).

## Requirements & Setup

Tested with Python 3.10+, PyTorch 2.11 (CUDA 12.8), NumPy 1.26, SciPy 1.16, and scikit-learn 1.6.

```bash
pip install -r requirements.txt
```

If using a CUDA GPU, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/). CPU execution is supported automatically when CUDA is unavailable.

## Running Experiments

Running the scripts below automatically creates a local `results/` directory and writes structured JSON outputs.

```bash
# Main suite: RQ1 (11 models x 3 datasets x 5 seeds), RQ2 (sparsity),
# RQ3 (0--30% noise), RQ4 (5 ablations), RQ5 (sensitivity), and RQ6 (efficiency)
python run_experiments.py

# Optional: run RQ5 hyperparameter sensitivity standalone with incremental saving
python run_rq5.py

# Additional analyses: RQ7 (uncertainty calibration & 2D PCA manifold),
# RQ8 (d, L, H, gamma_w), and RQ9 (convergence & multi-K cutoffs)
python run_additional_analyses.py

# Print formatted summary tables after running run_experiments.py
python summarize_results.py
```

### Default Hyperparameters & Baselines

- **Random seeds**: `{42, 1024, 2024, 3407, 7777}`
- **Shared defaults**: embedding size `d = 64`, GCN layers `L = 3`, batch size `1024`, Adam learning rate `1e-3`, L2 weight decay `1e-4`, max epochs `120`, early-stopping patience `10` checks (20 epochs)
- **DistSGCL defaults**: pre-training epochs `3`, GMM components `K = 8`, attention heads `H = 4`, temperature `tau = 0.2`, loss weights `lambda1 = 0.02`, `lambda2 = 0.02`, perturbation radius `eps_cl = 0.1`, Wasserstein penalty `gamma_w = 0.05`
- **Included baselines (`models.py`)**: MF-BPR, NCF, DiffNet, LightGCN, SGL, SEPT, SimGCL, HCCF, LightGCL, and XSimGCL

## Citation

If you find this code or benchmark setup useful in your research, please cite:

```bibtex
@article{zhao2026distsgcl,
  title   = {DistSGCL: Uncertainty-Aware Social Recommendation via Gaussian Distributional Pre-training and Graph Contrastive Learning},
  author  = {Zhao, Mengjie and Chen, Ying},
  journal = {Kybernetika},
  year    = {2026}
}
```

## License

The source code in this repository is released under the [MIT License](LICENSE). The benchmark datasets under `data/` remain subject to their original research terms noted in each dataset's `readme.txt`.
