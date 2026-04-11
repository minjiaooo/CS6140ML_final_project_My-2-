"""
mf_model.py - Matrix Factorization Baseline

Architecture:
    Each user and item is assigned one embedding vector.
    Predicted score = dot product of user vector and item vector.
    No hidden layers — this is the simplest possible recommendation model.

Ablation experiments supported via command-line arguments:
    --loss      : bpr (default) vs bce
    --embed_dim : 32 / 64 (default) / 128
    --reg_lambda: 1e-3 / 1e-4 (default) / 1e-5
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from dataset import build_dataloaders
from evaluate import evaluate
from plot_curves import plot_training_curves


# ──────────────────────────────────────────
# 1. Model
# ──────────────────────────────────────────

class MatrixFactorization(nn.Module):
    """
    Standard Matrix Factorization

    score(user, item) = user_embedding · item_embedding  (dot product)

    This is equivalent to linear regression:
        - item embedding acts as the weight vector
        - user embedding acts as the input features
    The model can only learn LINEAR relationships between users and items,
    which is its core limitation compared to Two-Tower with MLP layers.
    """

    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        # Xavier initialization: keeps initial values in a reasonable range
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        print(f"\nMatrix Factorization Model:")
        print(f"  embed_dim  = {embed_dim}")
        print(f"  n_users    = {n_users:,}")
        print(f"  n_items    = {n_items:,}")

    def forward(self, user_ids, pos_items, neg_items):
        """
        Training forward pass.
        Returns scores for positive and negative items.
        """
        u = self.user_emb(user_ids)
        pos = self.item_emb(pos_items)
        neg = self.item_emb(neg_items)
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        return pos_scores, neg_scores

    def get_user_vector(self, user_ids):
        return self.user_emb(user_ids)

    def get_item_vectors(self, item_ids):
        return self.item_emb(item_ids)


# ──────────────────────────────────────────
# 2. Loss Functions
# ──────────────────────────────────────────

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.

    Core idea: the score for a positive item should be HIGHER than
    the score for a negative item. We do not need exact probabilities,
    only the correct relative ranking.

    Formula: loss = -log(sigmoid(pos_score - neg_score))

    """

    def __init__(self, reg_lambda=1e-4):
        super().__init__()
        self.reg_lambda = reg_lambda

    def forward(self, pos_scores, neg_scores, model):
        bpr_loss = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-8
        ).mean()
        # L2 regularization on embeddings to prevent overfitting
        reg_loss = model.user_emb.weight.norm(2).pow(2) + model.item_emb.weight.norm(2).pow(2)
        return bpr_loss + self.reg_lambda * reg_loss


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss (ablation comparison).
    Treats each (user, item) pair as an independent binary classification.
    """

    def __init__(self, reg_lambda=1e-4):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pos_scores, neg_scores, model):
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        bce_loss = self.bce(scores, labels)
        reg_loss = model.user_emb.weight.norm(2).pow(2) + model.item_emb.weight.norm(2).pow(2)
        return bce_loss + self.reg_lambda * reg_loss


# ──────────────────────────────────────────
# 3. Training Loop
# ──────────────────────────────────────────

def train(config, output_dir="./results/mf"):
    os.makedirs(output_dir, exist_ok=True)
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader, val_loader, test_loader, n_users, n_items = build_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
    )

    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config["embed_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = BPRLoss(reg_lambda=config["reg_lambda"]) if config["loss"] == "bpr" \
           else BCELoss(reg_lambda=config["reg_lambda"])

    history = {"train_loss": [], "val_hr10": [], "val_ndcg10": []}
    best_hr10, best_ndcg10 = 0.0, 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Training | loss={config['loss']} | embed_dim={config['embed_dim']} | reg={config['reg_lambda']}")
    print("=" * 65)

    for epoch in range(1, config["n_epochs"] + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for users, pos_items, neg_items in train_loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_scores, neg_scores = model(users, pos_items, neg_items)
            loss = criterion(pos_scores, neg_scores, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, config["k_list"])
        val_hr10    = val_metrics["HR@10"]
        val_ndcg10  = val_metrics["NDCG@10"]

        history["train_loss"].append(train_loss)
        history["val_hr10"].append(val_hr10)
        history["val_ndcg10"].append(val_ndcg10)
        scheduler.step(val_hr10)

        print(
            f"Epoch {epoch:3d}/{config['n_epochs']} | "
            f"loss={train_loss:.4f} | "
            f"HR@10={val_hr10:.4f} | "
            f"NDCG@10={val_ndcg10:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_hr10 > best_hr10:
            best_hr10, best_ndcg10 = val_hr10, val_ndcg10
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_mf.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {best_epoch} | best HR@10={best_hr10:.4f}")
                break

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_mf.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, device, config["k_list"])

    print("\n" + "=" * 65)
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    plot_training_curves(history, output_dir, title=f"MF | loss={config['loss']} | dim={config['embed_dim']}")

    result = {
        "config":      config,
        "best_epoch":  best_epoch,
        "val_HR@10":   best_hr10,
        "val_NDCG@10": best_ndcg10,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return result


# ──────────────────────────────────────────
# 4. Main Entry Point
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Matrix Factorization Baseline")
    parser.add_argument("--data_dir",    type=str,   default="./data/processed")
    parser.add_argument("--results_dir", type=str,   default="./results/mf")
    parser.add_argument("--embed_dim",   type=int,   default=64)
    parser.add_argument("--loss",        type=str,   default="bpr", choices=["bpr", "bce"])
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--reg_lambda",  type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--n_epochs",    type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--k_list",      type=int,   nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    config = {
        "data_dir":   args.data_dir,
        "embed_dim":  args.embed_dim,
        "loss":       args.loss,
        "lr":         args.lr,
        "reg_lambda": args.reg_lambda,
        "batch_size": args.batch_size,
        "n_epochs":   args.n_epochs,
        "patience":   args.patience,
        "k_list":     args.k_list,
    }

    train(config, output_dir=args.results_dir)


if __name__ == "__main__":
    main()