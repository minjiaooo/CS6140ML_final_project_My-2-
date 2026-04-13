# Comparing Classical and Neural Models for Recommender Systems

CS6140 Machine Learning Final Project — Northeastern University

## Project Overview

Comparing Matrix Factorization (MF) and Two-Tower models for implicit feedback recommendation on Amazon Review datasets.

## Repository Structure

```
.
├── data_pipeline.py      # Data processing (JSON -> train/val/test pkl files)
├── dataset.py            # PyTorch Dataset & DataLoader (shared by MF and Two-Tower)
├── evaluate.py           # Evaluation metrics: HR@K, NDCG@K (shared)
├── plot_curves.py        # Training curve plotting (shared)
├── eda_plots.py          # EDA visualization (called by data_pipeline)
│
├── mf_model.py           # Matrix Factorization model + training
├── model.py              # Two-Tower model (UserTower + ItemTower + BPR Loss)
├── train.py              # Two-Tower training + ablation study
│
├── colab_notebook.ipynb  # Google Colab experiment runner
│
└── data/
    ├── processed/            # Musical Instruments (default)
    ├── processed_cds/        # CDs & Vinyl
    ├── processed_kindle/     # Kindle Store
    └── processed_movies/     # Movies & TV
```

## Quick Start

### 1. Process Data
```bash
python data_pipeline.py --input Musical_Instruments_5.json --output_dir ./data/processed
```

### 2. Train MF (best config)
```bash
python mf_model.py --lr 0.01 --reg_lambda 1e-6 --embed_dim 256 --n_epochs 300 --patience 10
```

### 3. Train Two-Tower (single run)
```bash
python train.py --lr 0.001 --reg_lambda 0.00001 --dropout 0.1 --n_epochs 100 --patience 15
```

### 4. Run Two-Tower Ablation Study
```bash
python train.py --ablation --lr 0.001 --reg_lambda 0.00001 --dropout 0.1 --n_epochs 100 --patience 15
```

## Models

### Matrix Factorization (MF)
- Architecture: `user_id -> Embedding -> Dot Product <- Embedding <- item_id`
- Loss: BPR (Bayesian Personalized Ranking) + L2 regularization
- Limitation: Only learns linear relationships between users and items
- Ablation dimensions:
  - Embedding dimension: 32 / 64 / 128 / 256 / 512
  - L2 regularization strength: 0 / 1e-6 / 1e-5 / 1e-4
  - Loss function: BPR vs BCE
  - Bias terms: with / without user & item bias

### Two-Tower
- Architecture: `user_id -> Embedding -> MLP -> L2 Norm -> Cosine Similarity <- L2 Norm <- MLP <- Embedding <- item_id`
- Loss: BPR + L2 regularization
- Regularization: BatchNorm + Dropout inside each Tower
- Advantage: Captures non-linear relationships via MLP; L2-normalized outputs enable ANN retrieval
- Ablation dimensions:
  - Activation function: ReLU / GELU / Tanh
  - MLP depth: 1 / 2 / 3 layers
  - Embedding dimension: 32 / 64 / 128

## Evaluation
- Protocol: 1 positive + 99 random negatives per user (He et al., 2017)
- Metrics: HR@K (Hit Ratio), NDCG@K (Normalized Discounted Cumulative Gain)
- K = 5, 10, 20

## Datasets
Amazon Review 5-core datasets:

| Dataset | Users | Items | Interactions | Avg/User |
|---------|-------|-------|-------------|----------|
| Musical Instruments | 24,780 | 9,930 | 156,681 | 6.3 |
| CDs & Vinyl | 107,546 | 71,943 | 1,161,916 | 10.8 |
| Kindle Store | 139,028 | 98,584 | 1,936,163 | 13.9 |
| Movies & TV | 281,514 | 59,067 | 2,652,975 | 9.4 |

## Results

### MF — Musical Instruments Dataset

#### Best Configuration

| Model | lr | reg | dim | loss | Best Epoch | HR@5 | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|---|---|---|---|---|
| MF (best) | 0.01 | 1e-6 | 256 | BPR | 120 | 0.3930 | 0.5136 | 0.3304 | 0.6431 |
| MF (baseline) | 0.001 | 1e-5 | 64 | BPR | 99 | 0.3393 | 0.4621 | 0.2842 | 0.6040 |

#### Ablation: Embedding Dimension (lr=0.01, reg=1e-6, BPR)

| dim | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| 32 | 157 | 0.4740 | 0.3020 |
| 64 | 146 | 0.4877 | 0.3126 |
| 128 | 128 | 0.4987 | 0.3217 |
| **256** | **120** | **0.5136** | **0.3304** |
| 512 | 91 | 0.5132 | 0.3299 |

#### Ablation: Regularization Strength (lr=0.01, dim=256, BPR)

| reg_lambda | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| 1e-4 | 89 | 0.3100 | 0.1878 |
| 1e-5 | 75 | 0.4636 | 0.2850 |
| **1e-6** | **120** | **0.5136** | **0.3304** |

#### Ablation: Loss Function (lr=0.01, reg=1e-6)

| loss | dim | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|---|
| **BPR** | 64 | 146 | **0.4877** | **0.3126** |
| BCE | 64 | 128 | 0.4768 | 0.3015 |
| **BPR** | 128 | 128 | **0.4987** | **0.3217** |
| BCE | 128 | 145 | 0.4918 | 0.3150 |
| **BPR** | 256 | 120 | **0.5136** | **0.3304** |
| BCE | 256 | 125 | 0.4970 | 0.3177 |

#### Ablation: Bias Terms (lr=0.01, reg=1e-6, dim=256, BPR)

| use_bias | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| **False** | **120** | **0.5136** | **0.3304** |
| True | 118 | 0.4761 | 0.3061 |

> Note: User bias cancels out in BPR pairwise difference (b_u appears in both pos and neg scores), making bias terms ineffective under BPR loss.

### Two-Tower — Musical Instruments Dataset (Preliminary)

| Model | Config | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|--------|------|-------|-------|--------|---------|---------|
| Two-Tower (preliminary) | ReLU, layers=2, dim=64, reg=1e-5 | 0.2967 | 0.7612* | 0.8397* | 0.1434 | 0.2953 | 0.3162 |

*\* Unstable due to random negative sampling in evaluation — to be re-evaluated after fixing seed.*

*Full ablation results in `results/` directory after fixing evaluation.*

## References
1. Koren et al. (2009) - Matrix Factorization Techniques for Recommender Systems
2. Rendle et al. (2009) - BPR: Bayesian Personalized Ranking from Implicit Feedback
3. Covington et al. (2016) - Deep Neural Networks for YouTube Recommendations
4. He et al. (2017) - Neural Collaborative Filtering
5. Ni et al. (2019) - Amazon Review Data (2018)
