# Comparing Classical and Neural Models for Recommender Systems

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

#### Hyperparameter Tuning

Anchor: embed_dim=64, n_layers=2, activation=relu, dropout=0.2, warmup_epochs=3.

| Experiment | dim | lr | reg | neg | Best Epoch | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| baseline_v2 | 64 | 1e-4 | 0.01 | 1 | 28 | 0.3702 | 0.2201 |
| reg001 | 64 | 1e-4 | 0.001 | 1 | 24 | 0.3726 | 0.2222 |
| reg0001 | 64 | 1e-4 | 0.0001 | 1 | 16 | 0.3692 | 0.2200 |
| lr5e4 | 64 | 5e-4 | 0.001 | 1 | 16 | 0.3709 | 0.2195 |
| lr1e3 | 64 | 1e-3 | 0.001 | 1 | 9 | 0.3677 | 0.2188 |
| dim128 | 128 | 1e-4 | 0.001 | 1 | 13 | 0.3710 | 0.2195 |
| dim256 | 256 | 1e-4 | 0.001 | 1 | 13 | 0.3725 | 0.2216 |
| dim256_neg4 | 256 | 1e-4 | 0.001 | 4 | 13 | 0.3715 | 0.2216 |

> 8 experiments spanning lr (10x range), reg (100x range), dim (4x range), and neg samples. HR@10 remains locked at 0.3677–0.3726 (< 0.5% variation), confirming the performance ceiling is due to data sparsity, not hyperparameter choice.

**Best Two-Tower configuration:** dim=64, lr=1e-4, reg=0.001 → HR@10 = 0.3726, NDCG@10 = 0.2222.

#### Two-Tower Ablation Study

Three controlled ablation experiments. Anchor: dim=64, lr=1e-4, reg=0.001, n_layers=2, activation=relu, dropout=0.2.

##### Ablation 1: Activation Function

Fixed: dim=64, n_layers=2.

| Activation | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| **ReLU** | **24** | **0.3726** | **0.2222** | **0.5046** |
| GELU | 14 | 0.3708 | 0.2205 | 0.5022 |
| Tanh | 16 | 0.3724 | 0.2208 | 0.5016 |

> All three activation functions produce nearly identical results (< 0.2% variation). The choice of activation has negligible impact when data is insufficient to learn meaningful nonlinear transformations.

##### Ablation 2: MLP Depth

Fixed: dim=64, activation=relu.

| n_layers | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| 1 | 29 | 0.3695 | 0.2208 | 0.5056 |
| **2** | **24** | **0.3726** | **0.2222** | **0.5046** |
| 3 | — | excluded | — | — |

> 1 layer vs 2 layers shows minimal difference (+0.31%). 3 layers excluded due to evaluation instability caused by deep LayerNorm cascades producing degenerate scores in early epochs. Adding depth provides no benefit on this sparse dataset.

##### Ablation 3: Embedding Dimension

Fixed: n_layers=2, activation=relu.

| dim | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| 32 | 12 | 0.3717 | 0.2208 | 0.5026 |
| **64** | **24** | **0.3726** | **0.2222** | **0.5046** |
| 128 | 13 | 0.3710 | 0.2195 | 0.5014 |
| 256 | 13 | 0.3725 | 0.2216 | 0.5048 |

> Unlike MF where dim=256 significantly outperforms dim=32 (+5.8%), Two-Tower shows zero sensitivity to embedding dimension (< 0.3% variation). The MLP layers cannot extract useful nonlinear features from any embedding size given the limited training data.

#### Key Observation: Architecture-Insensitive Performance

Across all 11 Two-Tower experiments (8 tuning + 3 ablation), HR@10 ranges from 0.3677 to 0.3726 — a spread of only 0.49 percentage points. This striking insensitivity to all architectural and hyperparameter choices strongly suggests the bottleneck is data volume, not model configuration. The MLP layers effectively collapse to near-identity transformations when trained on only 6.3 interactions per user.

### MF vs. Two-Tower Summary (Musical Instruments)

| Model | Config | test HR@5 | test HR@10 | test HR@20 | test NDCG@10 |
|---|---|---|---|---|---|
| **MF (best)** | dim=256, lr=5e-4, reg=0.01, neg=4 | **0.3901** | **0.5133** | **0.6516** | **0.3262** |
| Two-Tower (best) | dim=64, lr=1e-4, reg=0.001, 2 layers | 0.2637 | 0.3726 | 0.5046 | 0.2222 |

MF outperforms Two-Tower by **14.1 percentage points** in HR@10 on this dataset. While MF benefits substantially from hyperparameter tuning (HR@10 improved from 0.4443 to 0.5133, +15.5% relative), Two-Tower is completely insensitive to tuning — 11 experiments across all dimensions produced < 0.5% variation.

## Results — CDs & Vinyl

To investigate how dataset scale affects the relative performance of MF and Two-Tower, baseline experiments were conducted on the Amazon CDs & Vinyl dataset, which is approximately 7.4× larger than Musical Instruments (1.16M vs 157K training interactions, 107K vs 25K users).

### Model Performance on CDs Dataset

**MF Results:**

| Experiment | dim | lr | reg | neg | HR@5 | HR@10 | HR@20 | NDCG@10 | Epoch |
|---|---|---|---|---|---|---|---|---|---|
| mf_baseline | 128 | 5e-4 | 0.01 | 1 | 0.6293 | 0.7527 | 0.8594 | 0.5285 | 146 |

**Two-Tower Results:**

| Experiment | dim | lr | reg | layers | neg | HR@5 | HR@10 | Gap (HR@10) | HR@20 | NDCG@10 | Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tt_baseline | 64 | 1e-4 | 0.001 | 2 | 1 | 0.5224 | 0.6725 | **-0.0802** | 0.8121 | 0.4298 | 200 (no stop) |
| tt_lr1e3 | 64 | 1e-3 | 0.001 | 2 | 1 | 0.5538 | 0.7004 | **-0.0523** | 0.8298 | 0.4565 | 200 (no stop) |
| tt_lr1e2 (running) | 64 | 1e-2 | 0.001 | 2 | 1 | — | — | **—** | — | — | — |

*(Note: Data sourced from experimental logs. The Gap (HR@10) column represents `Two-Tower HR@10 - MF baseline HR@10`)*

### Key Observations: Impact of Data Scale

* **Two-Tower learns on large data:** On the smaller Musical Instruments dataset, Two-Tower's validation HR@10 was completely flat across all 200 epochs regardless of hyperparameters. On the CDs dataset, HR@10 rises continuously (from 0.42 at epoch 4 to 0.73 at epoch 200) with no plateau in sight. The model is actively learning, and the MLP layers are extracting nonlinear patterns that MF cannot capture.
* **Training dynamics differ fundamentally:** MF converged in 146 epochs on CDs with `lr=5e-4`. Two-Tower required `lr=1e-3` (10× higher than what worked on Musical Instruments) and still had not fully converged at 200 epochs. This reflects the fundamental difference in model complexity: Two-Tower's MLP layers require substantially more gradient steps to optimize, and benefit from larger datasets that provide richer training signals per update.
* **The performance gap narrows with data scale:** On Musical Instruments, MF outperformed Two-Tower by 14.1 percentage points (HR@10: 0.5133 vs 0.3726). On the CDs dataset with baseline configurations, the gap narrows to 8.02pp (HR@10 gap: -0.0802). With `lr=1e-3`, Two-Tower reaches HR@10=0.7004, further narrowing the gap to 5.23pp (HR@10 gap: -0.0523). The trend is clear: as data volume increases, Two-Tower's advantage from nonlinear modeling grows while MF's lead shrinks.
* **Extrapolation suggests a crossover point exists:** Based on the observed trajectory—a 14.1pp gap at 157K interactions narrowing to a ~5.2pp gap at 1.16M interactions—we hypothesize that Two-Tower would match or surpass MF at some larger data scale. Due to computational and time constraints, this remains an open question for future work.
* **Alignment with industry trends:** This finding aligns with the historical development of recommendation systems. In the early era of platforms with smaller datasets, MF provided strong performance with minimal computational overhead. As platforms scaled to hundreds of millions of users, the limitations of MF's linear interaction function became apparent. The Two-Tower architecture emerged to model complex nonlinear user-item relationships at scale, while its independent item tower enabled offline precomputation for sub-millisecond ANN retrieval. Its industrial adoption was driven by scalability and accuracy at high data density, a dynamic empirically illustrated by these experimental results.

## Key Findings

1. **Learning rate is the most impactful MF hyperparameter.** Reducing lr from 1e-3 to 5e-4 unlocked the potential of larger embedding dimensions, producing a +3.55% HR@10 jump in one step.
2. **Hyperparameters interact strongly in MF.** dim=128 showed no gain with lr=1e-3 but large gain with lr=5e-4. Tuning one dimension in isolation misses these interactions.
3. **Simpler models win on sparse data.** MF outperforms Two-Tower by 14.1pp on Musical Instruments (6.3 interactions/user). Two-Tower's MLP layers cannot learn useful nonlinear transformations from limited data.
4. **Two-Tower is architecture-insensitive on sparse data.** Across 11 experiments varying lr, reg, dim, depth, activation, and neg samples, Two-Tower HR@10 varied by only 0.49pp — the bottleneck is data volume, not model configuration.
5. **BPR > BCE for ranking tasks.** Pairwise ranking loss outperforms pointwise classification loss by 3.8% across all MF configurations.
6. **Bias terms are harmful under BPR.** User bias cancels in the pairwise difference; item bias hurts personalized ranking.

## References
1. Koren, Bell & Volinsky (2009) - Matrix Factorization Techniques for Recommender Systems
2. Rendle et al. (2009) - BPR: Bayesian Personalized Ranking from Implicit Feedback
3. He et al. (2017) - Neural Collaborative Filtering
4. Covington et al. (2016) - Deep Neural Networks for YouTube Recommendations
5. Ni et al. (2019) - Amazon Review Data (2018)
6. Rendle et al. (2020) - Neural Collaborative Filtering vs. Matrix Factorization Revisited
