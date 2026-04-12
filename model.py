"""
model.py - Two-Tower Model
CS6140 ML Final Project

Architecture:
    User Tower: user_id -> Embedding -> MLP -> L2 Norm
    Item Tower: item_id -> Embedding -> MLP -> L2 Norm
    Score = dot product (equivalent to cosine similarity after L2 Norm)

Ablation experiments supported:
    --activation : relu (default) / gelu / tanh
    --n_layers   : 1 / 2 (default) / 3
    --embed_dim  : 32 / 64 (default) / 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
# 1. Tower (shared structure for User and Item)
# ──────────────────────────────────────────

class Tower(nn.Module):
    """
    A single tower: MLP + L2 Normalization.

    Each layer consists of:
        Linear(embed_dim, embed_dim) -> BatchNorm1d -> Activation -> Dropout

    After MLP, L2 normalization projects the output onto a unit sphere,
    so that dot product = cosine similarity. This enables ANN retrieval
    in production systems.
    """

    def __init__(self, embed_dim: int, n_layers: int, activation: str, dropout: float):
        super().__init__()

        # Activation function factory (store class, not instance)
        act_dict = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation.lower() not in act_dict:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_dict.keys())}")
        act_fn = act_dict[activation.lower()]

        # Build MLP dynamically based on n_layers
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


# ──────────────────────────────────────────
# 2. Two-Tower Model
# ──────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model.

    User Tower and Item Tower are independent (no shared weights),
    which allows each tower to learn its own transformation.

    score(user, item) = dot product of L2-normalized vectors
                      = cosine similarity

    Compared to MF:
        MF:        user_emb · item_emb  (linear, no hidden layers)
        Two-Tower: Tower(user_emb) · Tower(item_emb)  (non-linear via MLP)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()

        # Embedding layers
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        # Xavier initialization: consistent with MF baseline
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Two independent towers
        self.user_tower = Tower(embed_dim, n_layers, activation, dropout)
        self.item_tower = Tower(embed_dim, n_layers, activation, dropout)

        print(f"\nTwo-Tower Model:")
        print(f"  embed_dim  = {embed_dim}")
        print(f"  n_layers   = {n_layers}")
        print(f"  activation = {activation}")
        print(f"  dropout    = {dropout}")
        print(f"  n_users    = {n_users:,}")
        print(f"  n_items    = {n_items:,}")

    def forward(self, user_ids, pos_items, neg_items):
        """
        Training forward pass.

        Args:
            user_ids  : (B,)
            pos_items : (B,)
            neg_items : (B, K)  K = n_neg_train (default: 1 during training)

        Returns:
            pos_scores : (B,)
            neg_scores : (B, K)
        """
        # User vector
        u_vec = self.user_tower(self.user_emb(user_ids))        # (B, embed_dim)

        # Positive item vector
        pos_vec = self.item_tower(self.item_emb(pos_items))     # (B, embed_dim)

        # Negative item vectors — reshape to pass through Tower (requires 2D input)
        B, K = neg_items.shape
        neg_vec = self.item_emb(neg_items)                      # (B, K, embed_dim)
        neg_vec = neg_vec.reshape(B * K, -1)                    # (B*K, embed_dim)
        neg_vec = self.item_tower(neg_vec)                      # (B*K, embed_dim)
        neg_vec = neg_vec.reshape(B, K, -1)                     # (B, K, embed_dim)

        # Dot product scores (= cosine similarity after L2 norm)
        pos_scores = torch.sum(u_vec * pos_vec, dim=1)                  # (B,)
        neg_scores = torch.sum(u_vec.unsqueeze(1) * neg_vec, dim=2)     # (B, K)

        return pos_scores, neg_scores

    def get_user_vector(self, user_ids):
        """Called by evaluate.py"""
        return self.user_tower(self.user_emb(user_ids))

    def get_item_vectors(self, item_ids):
        """Called by evaluate.py"""
        # item_ids can be (B,) or (B, K) — flatten to 2D, then restore shape
        original_shape = item_ids.shape
        item_ids_flat = item_ids.reshape(-1)  # (B*K,) or (B,)
        i_vec = self.item_emb(item_ids_flat)  # (B*K, embed_dim)
        i_vec = self.item_tower(i_vec)  # (B*K, embed_dim)
        return i_vec.reshape(*original_shape, -1).squeeze(-2) if item_ids.dim() > 1 \
            else i_vec
