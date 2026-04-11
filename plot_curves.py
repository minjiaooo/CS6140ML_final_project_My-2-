"""
plot_curves.py — Training Curve Plotting
CS6140 ML Final Project

Shared by both MF and Two-Tower models.
Plots training loss and validation metrics (HR@10, NDCG@10) over epochs.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, output_dir: str, title: str = ""):
    """
    Plot and save two figures to output_dir:
        (a) train loss over epochs
        (b) val HR@10 and NDCG@10 over epochs

    Parameters:
        history    : dict with keys "train_loss", "val_hr10", "val_ndcg10"
        output_dir : directory to save the plot
        title      : optional title for the loss curve (e.g. model config info)
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Loss curve
    axes[0].plot(epochs, history["train_loss"], color="#4A90D9", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(title if title else "Training Loss")
    axes[0].grid(True, alpha=0.3)

    # (b) Validation metrics curve
    axes[1].plot(epochs, history["val_hr10"],   label="HR@10",   color="#7ED321", linewidth=1.5)
    axes[1].plot(epochs, history["val_ndcg10"], label="NDCG@10", color="#F5A623", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {save_path}")
