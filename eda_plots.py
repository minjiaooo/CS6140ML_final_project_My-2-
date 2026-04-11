"""
eda_plots.py — EDA Visualization
CS6140 ML Final Project

Generate three EDA plots for the processed dataset:
    (a) User interaction count distribution
    (b) Item interaction count distribution
    (c) Interactions per year (temporal coverage)

Called from data_pipeline.py after processing, which provides the full
DataFrame with timestamp for temporal distribution.
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate three EDA plots and save to output_dir:
      (a) User interaction count distribution  — shows power-law tail
      (b) Item interaction count distribution  — same
      (c) Interactions per year                — temporal coverage
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) User interaction distribution (sorted descending, most active on the left)
    user_counts = df.groupby("user_idx").size().sort_values(ascending=False).reset_index(drop=True)
    axes[0].plot(range(len(user_counts)), user_counts.values, color="#4A90D9", linewidth=1)
    axes[0].set_xlabel("User rank")
    axes[0].set_ylabel("Number of interactions")
    axes[0].set_title("User interaction distribution")
    axes[0].grid(True, alpha=0.3)

    # (b) Item interaction distribution (sorted descending, most popular on the left)
    item_counts = df.groupby("item_idx").size().sort_values(ascending=False).reset_index(drop=True)
    axes[1].plot(range(len(item_counts)), item_counts.values, color="#F5A623", linewidth=1)
    axes[1].set_xlabel("Item rank")
    axes[1].set_ylabel("Number of interactions")
    axes[1].set_title("Item interaction distribution")
    axes[1].grid(True, alpha=0.3)

    # (c) Temporal distribution (interactions per year, line chart)
    year_counts = pd.to_datetime(df["timestamp"], unit="s").dt.year.value_counts().sort_index()
    axes[2].plot(year_counts.index, year_counts.values, color="#7ED321", linewidth=1.5, marker="o", markersize=4)
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Number of interactions")
    axes[2].set_title("Temporal distribution")
    for yr, cnt in year_counts.items():
        axes[2].text(yr, cnt + 100, str(cnt), ha="center", fontsize=7, rotation=45)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "eda_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"EDA plots saved to: {save_path}\n")

    # Print summary statistics for the report
    print("=== EDA Summary Statistics ===")
    print(f"  User interactions — mean: {user_counts.mean():.1f}, "
          f"median: {user_counts.median():.0f}, "
          f"max: {user_counts.max()}, "
          f"total users: {len(user_counts):,}")
    print(f"  Item interactions — mean: {item_counts.mean():.1f}, "
          f"median: {item_counts.median():.0f}, "
          f"max: {item_counts.max()}, "
          f"total items: {len(item_counts):,}")
    top10_pct = user_counts.iloc[:int(len(user_counts) * 0.1)].sum() / user_counts.sum()
    print(f"  Top 10% users account for {top10_pct:.1%} of all interactions (power-law)")
