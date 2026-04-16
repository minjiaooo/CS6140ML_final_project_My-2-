"""
train.py - Two-Tower Training + Ablation Study
CS6140 ML Final Project

Single run:
    python train.py --lr 0.0001 --reg_lambda 0.001 --dropout 0.2

Ablation study (loops over activation / n_layers / embed_dim):
    python train.py --ablation --lr 0.0001 --reg_lambda 0.001 --dropout 0.2
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from dataset import build_dataloaders
from evaluate import evaluate
from model import TwoTowerModel
from plot_curves import plot_training_curves


# ──────────────────────────────────────────
# 1. BPR Loss (same as MF baseline)
# ──────────────────────────────────────────

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.
    loss = -log(sigmoid(pos_score - neg_score))

    Uses batch-level L2 regularization on embeddings passed via emb_dict,
    consistent with MF baseline for fair comparison.
    """

    def __init__(self, reg_lambda: float = 1e-3):
        super().__init__()
        self.reg_lambda = reg_lambda

    def forward(self, pos_scores, neg_scores, emb_dict=None):
        # pos_scores: (B,)   neg_scores: (B,) or (B, K)
        if neg_scores.dim() > 1:
            pos_expanded = pos_scores.unsqueeze(1)                      # (B, 1)
            bpr_loss = -torch.log(
                torch.sigmoid(pos_expanded - neg_scores) + 1e-8
            ).mean()
        else:
            bpr_loss = -torch.log(
                torch.sigmoid(pos_scores - neg_scores) + 1e-8
            ).mean()

        # L2 regularization on BATCH embeddings (reuse from forward, no re-lookup)
        if self.reg_lambda > 0 and emb_dict is not None:
            B = emb_dict["u"].size(0)
            reg_loss = (
                emb_dict["u"].norm(2).pow(2)
                + emb_dict["pos"].norm(2).pow(2)
                + emb_dict["neg"].norm(2).pow(2)
            ) / B
            return bpr_loss + self.reg_lambda * reg_loss
        return bpr_loss


# ──────────────────────────────────────────
# 2. Training Loop
# ──────────────────────────────────────────

def train(config: dict, output_dir: str = "./results/two_tower"):
    os.makedirs(output_dir, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, n_users, n_items = build_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        n_neg_train=config.get("n_neg_train", 1),
    )

    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        activation=config["activation"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = BPRLoss(reg_lambda=config["reg_lambda"])

    history = {"train_loss": [], "val_hr10": [], "val_ndcg10": []}
    best_hr10, best_ndcg10 = 0.0, 0.0
    patience_counter = 0
    best_epoch = 0

    tag = f"act={config['activation']} | layers={config['n_layers']} | dim={config['embed_dim']}"
    print(f"\nTraining | {tag}")
    print("=" * 65)

    for epoch in range(1, config["n_epochs"] + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for users, pos_items, neg_items in train_loader:
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)   # (B, n_neg) from dataset

            pos_scores, neg_scores, emb_dict = model(users, pos_items, neg_items)
            loss = criterion(pos_scores, neg_scores, emb_dict=emb_dict)

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

        if epoch <= config.get("warmup_epochs", 3):
            # Warmup: train but don't track best model
            continue

        if val_hr10 > best_hr10:
            best_hr10, best_ndcg10 = val_hr10, val_ndcg10
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_two_tower.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {best_epoch} | best HR@10={best_hr10:.4f}")
                break

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_two_tower.pt"), map_location=device)
    )
    test_metrics = evaluate(model, test_loader, device, config["k_list"])

    print("\n" + "=" * 65)
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    plot_training_curves(
        history, output_dir,
        title=f"Two-Tower | {tag}"
    )

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
# 3. Ablation Study
# ──────────────────────────────────────────

def run_ablation(base_config: dict, results_root: str = "./results/two_tower_ablation"):
    """
    Ablation over three dimensions:
        - activation : relu / gelu / tanh
        - n_layers   : 1 / 2 / 3
        - embed_dim  : 32 / 64 / 128

    One dimension is varied at a time; the others are fixed to base_config values.
    Results are saved to separate subdirectories and summarized in ablation_summary.json.
    """
    ablation_grid = {
        "activation": ["relu", "gelu", "tanh"],
        "n_layers":   [1, 2, 3],
        "embed_dim":  [32, 64, 128],
    }

    all_results = []

    for dim, values in ablation_grid.items():
        for val in values:
            config = base_config.copy()
            config[dim] = val

            run_name = f"{dim}={val}"
            output_dir = os.path.join(results_root, run_name)
            print(f"\n{'='*65}")
            print(f"Ablation: {run_name}")
            print(f"{'='*65}")

            result = train(config, output_dir=output_dir)
            result["ablation_dim"] = dim
            result["ablation_val"] = str(val)
            all_results.append(result)

    # Save summary
    os.makedirs(results_root, exist_ok=True)
    summary_path = os.path.join(results_root, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Run':<20} {'HR@10':>8} {'NDCG@10':>10} {'test_HR@10':>12} {'test_NDCG@10':>14}")
    print("-" * 65)
    for r in all_results:
        run_name = f"{r['ablation_dim']}={r['ablation_val']}"
        print(
            f"{run_name:<20} "
            f"{r['val_HR@10']:>8.4f} "
            f"{r['val_NDCG@10']:>10.4f} "
            f"{r['test_HR@10']:>12.4f} "
            f"{r['test_NDCG@10']:>14.4f}"
        )
    print(f"\nAblation summary saved to {summary_path}")
    return all_results


# ──────────────────────────────────────────
# 4. Main Entry Point
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Two-Tower Training")
    parser.add_argument("--data_dir",    type=str,   default="./data/processed")
    parser.add_argument("--results_dir", type=str,   default="./results/two_tower")
    parser.add_argument("--embed_dim",   type=int,   default=64)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--activation",  type=str,   default="relu",
                        choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--reg_lambda",  type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--n_epochs",    type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--k_list",      type=int,   nargs="+", default=[5, 10, 20])
    parser.add_argument("--n_neg_train", type=int,   default=1,
                        help="Number of negative samples per positive during training")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Number of warmup epochs before tracking best model")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run full ablation study")
    args = parser.parse_args()

    config = {
        "data_dir":      args.data_dir,
        "embed_dim":     args.embed_dim,
        "n_layers":      args.n_layers,
        "activation":    args.activation,
        "dropout":       args.dropout,
        "lr":            args.lr,
        "reg_lambda":    args.reg_lambda,
        "batch_size":    args.batch_size,
        "n_epochs":      args.n_epochs,
        "patience":      args.patience,
        "k_list":        args.k_list,
        "n_neg_train":   args.n_neg_train,
        "warmup_epochs": args.warmup_epochs,
    }

    if args.ablation:
        run_ablation(config, results_root=args.results_dir + "_ablation")
    else:
        train(config, output_dir=args.results_dir)


if __name__ == "__main__":
    main()
