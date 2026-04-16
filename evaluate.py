"""
evaluate.py — Evaluation Metrics and Functions
CS6140 ML Final Project

Shared by both MF and Two-Tower models.

Metrics:
    HR@K  — Hit Ratio: is the ground truth in the top-K?
    NDCG@K — Normalized DCG: how high is the ground truth ranked?

Evaluation protocol:
    1 positive + 99 negative samples per user (NCF, He et al. 2017)
"""

import numpy as np
import torch


def hit_ratio_at_k(scores, k):
    """
    HR@K: fraction of users whose ground truth item appears in top-K.
    scores shape: (n_users, 100) — column 0 is always the ground truth.
    """
    # Sort each user's 100 scores from high to low
    sorted_indices = np.argsort(-scores, axis=1)
    # Keep only the top-K indices
    top_k = sorted_indices[:, :k]
    # Check if the ground truth (column 0) appears in top-K for each user
    hits = []
    for i in range(len(top_k)):
        if 0 in top_k[i]:
            hits.append(1.0)        # yes → hit
        else:
            hits.append(0.0)        # no → miss
    # Average across all users
    return np.mean(hits)


def ndcg_at_k(scores, k):
    """
    NDCG@K: like HR@K but rewards models that rank the ground truth higher.
    NDCG = 1 / log2(rank + 1)
    A ground truth at rank 1 scores 1.0; at rank 2 scores ~0.63; etc.
    """
    # Sort each user's 100 scores from high to low, get column indices
    sorted_indices = np.argsort(-scores, axis=1)
    # Keep only the top-K indices
    top_k = sorted_indices[:, :k]
    ndcgs = []
    for i in range(len(top_k)):
        # Find the position of ground truth (column 0) in this user's top-K
        found = False
        for j in range(k):
            if top_k[i][j] == 0:        # ground truth found at position j
                rank = j + 1             # convert to 1-indexed rank
                ndcgs.append(1.0 / np.log2(rank + 1))
                found = True
                break
        if not found:
            ndcgs.append(0.0)            # ground truth not in top-K
    return np.mean(ndcgs)


@torch.no_grad()
def evaluate(model, loader, device, k_list=[5, 10, 20]):
    """
    Compute HR@K and NDCG@K on val or test set.

    For each user, compute scores for 1 positive + 99 negative samples,
    then check where the positive sample ranks among the 100 candidates.

    Returns: dict, e.g. {"HR@5": 0.42, "NDCG@5": 0.31, "HR@10": 0.55, ...}
    """
    model.eval()
    all_scores = []
    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        u_vec = model.get_user_vector(users)                        # (B, dim)

        # Combine pos and neg items, pass through item tower TOGETHER
        # so that LayerNorm sees the same batch statistics for both
        all_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # (B, 100)
        B, N = all_items.shape
        all_items_flat = all_items.reshape(-1)                      # (B*100,)
        all_vec = model.get_item_vectors(all_items_flat)            # (B*100, dim)
        all_vec = all_vec.reshape(B, N, -1)                         # (B, 100, dim)

        # Scores: column 0 = positive, columns 1-99 = negatives
        batch_scores = (u_vec.unsqueeze(1) * all_vec).sum(dim=2)    # (B, 100)
        all_scores.append(batch_scores.cpu().numpy())

    all_scores = np.vstack(all_scores)   # (n_users, 100)

    metrics = {}
    for k in k_list:
        metrics[f"HR@{k}"]   = hit_ratio_at_k(all_scores, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(all_scores, k)
    return metrics
