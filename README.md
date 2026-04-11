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

### 2. Train MF (baseline)
```bash
python mf_model.py --lr 0.001 --reg_lambda 1e-5 --n_epochs 100
```

### 3. Train Two-Tower
```bash
python train.py --lr 0.0001 --reg_lambda 0.001 --dropout 0.2
```

### 4. Run Two-Tower Ablation Study
```bash
python train.py --ablation --lr 0.0001 --reg_lambda 0.001 --dropout 0.2
```

## Models

### Matrix Factorization (MF)
- Architecture: `user_id -> Embedding -> Dot Product <- Embedding <- item_id`
- Loss: BPR (Bayesian Personalized Ranking) + L2 regularization
- Only learns linear relationships between users and items

### Two-Tower
- Architecture: `user_id -> Embedding -> MLP -> L2 Norm -> Cosine Similarity <- L2 Norm <- MLP <- Embedding <- item_id`
- Loss: BPR + L2 regularization + Dropout + BatchNorm
- Can learn non-linear relationships via MLP layers

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

## References
1. Koren et al. (2009) - Matrix Factorization Techniques for Recommender Systems
2. Rendle et al. (2009) - BPR: Bayesian Personalized Ranking from Implicit Feedback
3. Covington et al. (2016) - Deep Neural Networks for YouTube Recommendations
4. He et al. (2017) - Neural Collaborative Filtering
5. Ni et al. (2019) - Amazon Review Data (2018)
