# Two-Tower Hyperparameter Tuning Report

**From Initial Configuration to Final Results**

CS6140 Machine Learning Final Project — Amazon Musical Instruments Dataset

---

## 1. Tuning Overview

This document records the complete hyperparameter tuning and ablation study for the Two-Tower model. Unlike the MF tuning process which showed significant sensitivity to hyperparameters, the Two-Tower model on this sparse dataset exhibits a striking insensitivity to all architectural and hyperparameter choices — a finding that is itself a key result of this study.

**Dataset:** Amazon Musical Instruments (5-core). 24,780 users, 9,930 items, 156,681 training interactions (6.3 per user on average).
**Evaluation:** Leave-one-out with 1 positive + 99 negatives, metrics HR@K and NDCG@K at K=5, 10, 20.
**Model:** Two-Tower with independent User and Item towers, each consisting of Linear → LayerNorm → Activation → Dropout layers. Trained with BPR loss and batch-level L2 regularization (same as MF for fair comparison).

---

## 2. Phase 1: Regularization Search

We start with a standard configuration: embed_dim=64, lr=1e-4, n_layers=2, activation=relu, dropout=0.2, warmup_epochs=3. The first variable explored is regularization strength.

| Experiment | dim | lr | reg | neg | HR@10 | NDCG@10 | epoch |
|---|---|---|---|---|---|---|---|
| tt_reg01 | 64 | 1e-4 | 0.01 | 1 | 0.3702 | 0.2201 | 28 |
| **tt_reg001** | **64** | **1e-4** | **0.001** | **1** | **0.3726** | **0.2222** | **24** |
| tt_reg0001 | 64 | 1e-4 | 0.0001 | 1 | 0.3692 | 0.2200 | 16 |

*Table 1. Phase 1 results: regularization search. Bold = phase best.*

**Observations:** All three regularization strengths produce nearly identical results (HR@10 range: 0.3692–0.3726, spread = 0.34%). Unlike MF where reg had a measurable impact, Two-Tower is insensitive to regularization — likely because dropout=0.2 already provides sufficient regularization for the MLP layers.

→ **Decision:** reg=0.001 marginally best. Explore learning rate next.

---

## 3. Phase 2: Learning Rate Search

We test whether a larger learning rate helps the MLP layers converge to better solutions. Two-Tower has more parameters than MF (MLP weights on top of embeddings), which may require different lr dynamics.

| Experiment | dim | lr | reg | neg | HR@10 | NDCG@10 | epoch |
|---|---|---|---|---|---|---|---|
| tt_lr1e4 (baseline) | 64 | 1e-4 | 0.001 | 1 | 0.3726 | 0.2222 | 24 |
| tt_lr5e4 | 64 | 5e-4 | 0.001 | 1 | 0.3709 | 0.2195 | 16 |
| tt_lr1e3 | 64 | 1e-3 | 0.001 | 1 | 0.3677 | 0.2188 | 9 |

*Table 2. Phase 2 results: learning rate search. Bold = phase best.*

**Observations:** Increasing lr does not improve performance — it slightly worsens it (0.3726 → 0.3677). Higher lr also causes faster early stopping (24 → 9 epochs), suggesting the model quickly overfits without learning better representations. This is the opposite of MF, where reducing lr from 1e-3 to 5e-4 was the single biggest breakthrough.

→ **Decision:** lr=1e-4 remains optimal. The bottleneck is not convergence speed. Explore model capacity (dim) next.

---

## 4. Phase 3: Embedding Dimension and Negative Sampling

We test whether larger embeddings or more negative samples can push Two-Tower beyond the 0.37 ceiling.

| Experiment | dim | lr | reg | neg | HR@10 | NDCG@10 | epoch |
|---|---|---|---|---|---|---|---|
| tt_dim64 (baseline) | 64 | 1e-4 | 0.001 | 1 | 0.3726 | 0.2222 | 24 |
| tt_dim128 | 128 | 1e-4 | 0.001 | 1 | 0.3710 | 0.2195 | 13 |
| tt_dim256 | 256 | 1e-4 | 0.001 | 1 | 0.3725 | 0.2216 | 13 |
| tt_dim256_neg4 | 256 | 1e-4 | 0.001 | 4 | 0.3715 | 0.2216 | 13 |

*Table 3. Phase 3 results: dimension and negative sampling search.*

**Observations:** Neither larger embeddings (dim 64 → 128 → 256) nor more negative samples (neg 1 → 4) produce any improvement. HR@10 remains locked at 0.37 regardless of model capacity. This confirms the performance ceiling is not a capacity limitation — the MLP layers simply cannot learn useful nonlinear transformations from 6.3 interactions per user.

→ **Decision:** dim=64 is sufficient. The bottleneck is data volume, not model size. Proceed to ablation study on architecture choices.

---

## 5. Ablation Study

Three controlled ablation experiments using the best configuration as anchor: dim=64, lr=1e-4, reg=0.001, n_layers=2, activation=relu, dropout=0.2.

### Ablation 1: Activation Function

Fixed: dim=64, n_layers=2, lr=1e-4, reg=0.001.

| Activation | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| **ReLU** | **24** | **0.3726** | **0.2222** | **0.5046** |
| GELU | 14 | 0.3708 | 0.2205 | 0.5022 |
| Tanh | 16 | 0.3724 | 0.2208 | 0.5016 |

*Table 4. Activation function ablation. Bold = best.*

**Observations:** All three activations produce virtually identical results (< 0.2% variation). ReLU is marginally best and converges most slowly (24 epochs vs 14–16), suggesting it extracts slightly more signal before plateauing. The insensitivity to activation function confirms that the MLP layers are not learning meaningful nonlinear transformations — they effectively collapse to near-linear mappings regardless of the activation choice.

### Ablation 2: MLP Depth

Fixed: dim=64, activation=relu, lr=1e-4, reg=0.001.

| n_layers | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| 1 | 29 | 0.3695 | 0.2208 | 0.5056 |
| **2** | **24** | **0.3726** | **0.2222** | **0.5046** |
| 3 | — | excluded | — | — |

*Table 5. MLP depth ablation. Bold = best. 3 layers excluded due to evaluation instability.*

**Observations:** 1 layer vs 2 layers shows minimal difference (+0.31% HR@10). 3 layers is excluded because deep LayerNorm cascades produce degenerate evaluation scores in early training epochs, causing the warmup mechanism to save an unreliable checkpoint. The lack of improvement from added depth reinforces the conclusion that the MLP layers are not contributing useful non-linear capacity on this dataset.

### Ablation 3: Embedding Dimension

Fixed: n_layers=2, activation=relu, lr=1e-4, reg=0.001.

| dim | Best Epoch | HR@10 | NDCG@10 | HR@20 |
|---|---|---|---|---|
| 32 | 12 | 0.3717 | 0.2208 | 0.5026 |
| **64** | **24** | **0.3726** | **0.2222** | **0.5046** |
| 128 | 13 | 0.3710 | 0.2195 | 0.5014 |
| 256 | 13 | 0.3725 | 0.2216 | 0.5048 |

*Table 6. Embedding dimension ablation. Bold = best.*

**Observations:** Unlike MF where dim=256 significantly outperforms dim=32 by +5.8% HR@10, Two-Tower shows zero sensitivity to embedding dimension (< 0.3% variation across 8x range). The MLP layers cannot extract useful nonlinear features from any embedding size given the limited training data. This is a sharp contrast to MF's behavior and highlights how data sparsity differentially affects linear vs. nonlinear models.

---

## 6. Complete Results Table

All 11 experiments (8 tuning + 3 ablation):

| Phase | Experiment | dim | lr | reg | neg | layers | activation | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tt_reg01 | 64 | 1e-4 | 0.01 | 1 | 2 | relu | 0.2613 | 0.3702 | 0.5027 | 0.1850 | 0.2201 | 0.2534 | 28 |
| 1 | **tt_reg001** | **64** | **1e-4** | **0.001** | **1** | **2** | **relu** | **0.2637** | **0.3726** | **0.5046** | **0.1871** | **0.2222** | **0.2555** | **24** |
| 1 | tt_reg0001 | 64 | 1e-4 | 0.0001 | 1 | 2 | relu | 0.2628 | 0.3692 | 0.5015 | 0.1857 | 0.2200 | 0.2533 | 16 |
| 2 | tt_lr5e4 | 64 | 5e-4 | 0.001 | 1 | 2 | relu | 0.2622 | 0.3709 | 0.5054 | 0.1844 | 0.2195 | 0.2534 | 16 |
| 2 | tt_lr1e3 | 64 | 1e-3 | 0.001 | 1 | 2 | relu | 0.2573 | 0.3677 | 0.5000 | 0.1831 | 0.2188 | 0.2521 | 9 |
| 3 | tt_dim128 | 128 | 1e-4 | 0.001 | 1 | 2 | relu | 0.2601 | 0.3710 | 0.5014 | 0.1838 | 0.2195 | 0.2523 | 13 |
| 3 | tt_dim256 | 256 | 1e-4 | 0.001 | 1 | 2 | relu | 0.2630 | 0.3725 | 0.5048 | 0.1863 | 0.2216 | 0.2549 | 13 |
| 3 | tt_dim256_neg4 | 256 | 1e-4 | 0.001 | 4 | 2 | relu | 0.2626 | 0.3715 | 0.5042 | 0.1865 | 0.2216 | 0.2550 | 13 |
| A1 | tt_gelu | 64 | 1e-4 | 0.001 | 1 | 2 | gelu | 0.2622 | 0.3708 | 0.5022 | 0.1855 | 0.2205 | 0.2536 | 14 |
| A1 | tt_tanh | 64 | 1e-4 | 0.001 | 1 | 2 | tanh | 0.2616 | 0.3724 | 0.5016 | 0.1851 | 0.2208 | 0.2534 | 16 |
| A2 | tt_1layer | 64 | 1e-4 | 0.001 | 1 | 1 | relu | 0.2629 | 0.3695 | 0.5056 | 0.1863 | 0.2208 | 0.2551 | 29 |
| A3 | tt_dim32 | 32 | 1e-4 | 0.001 | 1 | 2 | relu | 0.2606 | 0.3717 | 0.5026 | 0.1850 | 0.2208 | 0.2538 | 12 |

*Bold = overall best configuration. A1/A2/A3 = ablation experiments.*

---

## 7. Summary: Architecture-Insensitive Performance

Across all 11 experiments, HR@10 ranges from **0.3677 to 0.3726** — a total spread of only **0.49 percentage points**. This is in stark contrast to MF, where tuning improved HR@10 from 0.4443 to 0.5133 (+15.5% relative).

| Dimension Varied | Range Tested | HR@10 Spread | Impact |
|---|---|---|---|
| Regularization (reg) | 0.0001 – 0.01 (100x) | 0.34% | Negligible |
| Learning rate (lr) | 1e-4 – 1e-3 (10x) | 0.49% | Negligible |
| Embedding dim | 32 – 256 (8x) | 0.31% | Negligible |
| Negative samples | 1 – 4 (4x) | 0.11% | Negligible |
| Activation function | ReLU / GELU / Tanh | 0.18% | Negligible |
| MLP depth | 1 – 2 layers | 0.31% | Negligible |

*Table 7. Sensitivity analysis: HR@10 variation across all dimensions.*

---

## 8. Key Takeaways

1. **Two-Tower hits a hard ceiling on sparse data.** With only 6.3 training interactions per user, the MLP layers cannot learn meaningful nonlinear transformations. The model effectively degenerates to a noisy version of MF.

2. **All hyperparameters and architecture choices are irrelevant.** Unlike MF where lr, dim, and reg each had measurable effects, Two-Tower performance is completely invariant to tuning — the bottleneck is data volume, not model configuration.

3. **Deeper is not better on sparse data.** Adding MLP layers (1 → 2) provides only +0.31% improvement. The additional parameters introduce optimization complexity without access to sufficient training signal.

4. **Contrast with MF is the key finding.** MF benefits substantially from tuning (+15.5% relative improvement) because its simple linear structure can be precisely optimized with limited data. Two-Tower's nonlinear layers add noise rather than signal in this regime.

---

**Best Two-Tower Configuration:**

```
embed_dim=64, lr=1e-4, reg_lambda=0.001, n_layers=2, activation=relu, dropout=0.2
Test HR@10 = 0.3726, Test NDCG@10 = 0.2222
```

**Comparison with MF:**

| Model | test HR@10 | test NDCG@10 | Tuning Sensitivity |
|---|---|---|---|
| **MF (best)** | **0.5133** | **0.3262** | High (+15.5% from tuning) |
| Two-Tower (best) | 0.3726 | 0.2222 | None (< 0.5% variation) |

MF outperforms Two-Tower by **14.1 percentage points** in HR@10 on this sparse dataset.
