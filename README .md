# Comparing Classical and Neural Models for Recommender Systems

CS6140 Machine Learning Final Project — Northeastern University

## Project Overview

Comparing Matrix Factorization (MF) and Two-Tower models for implicit feedback recommendation on Amazon Review datasets. Both models share the same data pipeline, evaluation protocol, and loss function implementation, ensuring a fair controlled comparison.

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
├── model.py              # Two-Tower model architecture
├── train.py              # Two-Tower training + ablation study
│
└── data/
    ├── processed_musical/    # Amazon Musical Instruments (small)
    └── processed_cds/        # Amazon CDs & Vinyl (large)
```

## Quick Start

### 1. Process Data
```bash
# Musical Instruments
python data_pipeline.py --input Musical_Instruments_5.json --output_dir ./data/processed_musical

# CDs & Vinyl
python data_pipeline.py --input CDs_and_Vinyl_5.json --output_dir ./data/processed_cds
```

### 2. Train MF (best config)
```bash
python mf_model.py \
  --data_dir ./data/processed_musical \
  --embed_dim 256 --lr 5e-4 --reg_lambda 0.01 --n_neg_train 4 \
  --batch_size 1024 --n_epochs 200 --patience 10
```

### 3. Train Two-Tower
```bash
python train.py \
  --data_dir ./data/processed_musical \
  --embed_dim 64 --lr 1e-4 --reg_lambda 0.001 \
  --n_layers 2 --activation relu --dropout 0.2 \
  --n_neg_train 1 --warmup_epochs 3 \
  --batch_size 1024 --n_epochs 200 --patience 10
```

## Models

### Matrix Factorization (MF)
- Architecture: `user_id → Embedding → Dot Product ← Embedding ← item_id`
- Loss: BPR (Bayesian Personalized Ranking) + batch-level L2 regularization
- Regularization: L2 on batch embeddings (penalizes only embeddings used in current batch)
- Limitation: Only learns linear (bilinear) relationships between users and items

### Two-Tower
- Architecture: `user_id → Embedding → MLP (Linear→LayerNorm→ReLU→Dropout) × N → Dot Product ← MLP × N ← Embedding ← item_id`
- Loss: BPR + batch-level L2 regularization (same as MF for fair comparison)
- Regularization: LayerNorm + Dropout inside each Tower, L2 on batch embeddings
- Advantage: Captures non-linear relationships via MLP layers

### Key Design: Unified Loss Interface
Both models return `(pos_scores, neg_scores, emb_dict)` from forward(). The `emb_dict` contains pre-computed embeddings reused for L2 regularization, avoiding redundant embedding lookups and ensuring identical regularization behavior across models.

## Evaluation
- Protocol: Leave-one-out with 1 positive + 99 random negatives per user (He et al., 2017)
- Negative samples fixed with random seed 42 for reproducibility
- Metrics: HR@K (Hit Ratio), NDCG@K (Normalized Discounted Cumulative Gain)
- K = 5, 10, 20

## Datasets

| Dataset | Users | Items | Train Interactions | Avg/User |
|---------|-------|-------|-------------------|----------|
| Musical Instruments | 24,780 | 9,930 | 156,681 | 6.3 |
| CDs & Vinyl | 107,546 | 71,943 | 1,161,916 | 10.8 |

## Results — Musical Instruments

### MF Best Configuration

| Model | dim | lr | reg | neg | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| **MF (best)** | **256** | **5e-4** | **0.01** | **4** | **79** | **0.5133** | **0.3262** |
| MF (baseline) | 64 | 1e-3 | 1e-5 | 1 | 28 | 0.4443 | 0.2776 |

### MF Hyperparameter Tuning Trajectory

| Phase | Key Change | HR@10 | Δ | Insight |
|---|---|---|---|---|
| Baseline | dim=64, lr=1e-3, reg=1e-5 | 0.4443 | — | Starting point |
| Phase 1 | reg: 1e-5 → 1e-3 | 0.4479 | +0.36% | Batch reg needs larger λ |
| Phase 2 | dim: 64 → 128 | 0.4472 | +0.29% | Larger dim helps, but lr limits it |
| **Phase 3** | **lr: 1e-3 → 5e-4, dim=128** | **0.4834** | **+3.55%** | **Breakthrough: slower lr unlocks dim** |
| **Phase 4** | **dim=256, reg=0.01, neg=4** | **0.5133** | **+2.99%** | **Scale + neg sampling + reg balance** |

Total improvement: HR@10 0.4443 → 0.5133 (+6.9pp, +15.5% relative)

### MF Ablation Study

Anchor configuration: dim=256, BPR, lr=5e-4, reg=0.01, n_neg=4, no bias.

#### Ablation 1: Loss Function (BPR vs. BCE)

| Loss | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| **BPR** | **79** | **0.5133** | **0.3262** |
| BCE | 166 | 0.4944 | 0.3023 |

> BPR outperforms BCE by 3.8% in HR@10. BPR directly optimizes pairwise ranking, while BCE treats each pair as independent binary classification.

#### Ablation 2: Embedding Dimension

| dim | Best Epoch | HR@10 | NDCG@10 | Δ HR@10 |
|---|---|---|---|---|
| 32 | 112 | 0.4853 | 0.3055 | — |
| 64 | 124 | 0.4980 | 0.3148 | +1.27% |
| 128 | 94 | 0.5081 | 0.3222 | +1.01% |
| **256** | **79** | **0.5133** | **0.3262** | **+0.52%** |

> Monotonic improvement with diminishing returns. dim=256 is the effective capacity limit for this dataset.

#### Ablation 3: Negative Sampling Ratio

| n_neg | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| 1 | 108 | 0.5055 | 0.3220 |
| **4** | **79** | **0.5133** | **0.3262** |

> n_neg=4 provides +0.78% HR@10 with faster convergence. Modest gain consistent with He et al. findings on medium-scale datasets.

#### Ablation 4: Bias Terms

| use_bias | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|
| **False** | **79** | **0.5133** | **0.3262** |
| True | 59 | 0.5047 | 0.3212 |

> Bias terms hurt performance under BPR: user bias cancels in pairwise difference (ŷ_ui − ŷ_uj), and item bias introduces a popularity prior that interferes with personalized ranking.

### Two-Tower — Musical Instruments

| lr | reg | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|---|
| 1e-4 | 0.01 | 28 | 0.3702 | 0.2201 |
| 1e-4 | 0.001 | 24 | 0.3726 | 0.2222 |
| 1e-4 | 0.0001 | 16 | 0.3692 | 0.2200 |
| 5e-4 | 0.001 | 16 | 0.3709 | 0.2195 |
| 1e-3 | 0.001 | 9 | 0.3677 | 0.2188 |

> Two-Tower performance is stable at HR@10 ≈ 0.37 across all hyperparameter settings, significantly below MF's 0.5133. On this small, sparse dataset (avg 6.3 interactions/user), the MLP layers introduce parameters that cannot be learned from limited data, while MF's simpler linear structure is more effective.

### MF vs. Two-Tower Summary (Musical Instruments)

| Model | test HR@5 | test HR@10 | test HR@20 | test NDCG@10 |
|---|---|---|---|---|
| **MF (best)** | **0.3901** | **0.5133** | **0.6516** | **0.3262** |
| Two-Tower (best) | 0.2637 | 0.3726 | 0.5046 | 0.2222 |

MF outperforms Two-Tower by **14.1 percentage points** in HR@10 on this dataset.

## Results — CDs & Vinyl (In Progress)

Experiments running on the larger CDs dataset to test whether increased data volume allows Two-Tower to close the gap with MF.

## Key Findings

1. **Learning rate is the most impactful MF hyperparameter.** Reducing lr from 1e-3 to 5e-4 unlocked the potential of larger embedding dimensions, producing a +3.55% HR@10 jump in one step.
2. **Hyperparameters interact strongly.** dim=128 showed no gain with lr=1e-3 but large gain with lr=5e-4. Tuning one dimension in isolation misses these interactions.
3. **Simpler models win on sparse data.** MF consistently outperforms Two-Tower on Musical Instruments, where users average only 6.3 training interactions.
4. **BPR > BCE for ranking tasks.** Pairwise ranking loss outperforms pointwise classification loss across all configurations.
5. **Bias terms are harmful under BPR.** User bias cancels in the pairwise difference; item bias hurts personalized ranking.

## References
1. Koren, Bell & Volinsky (2009) - Matrix Factorization Techniques for Recommender Systems
2. Rendle et al. (2009) - BPR: Bayesian Personalized Ranking from Implicit Feedback
3. He et al. (2017) - Neural Collaborative Filtering
4. Covington et al. (2016) - Deep Neural Networks for YouTube Recommendations
5. Ni et al. (2019) - Amazon Review Data (2018)
6. Rendle et al. (2020) - Neural Collaborative Filtering vs. Matrix Factorization Revisited
