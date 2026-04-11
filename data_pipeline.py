"""
Data Pipeline for 5-core datasets

Pipeline steps:
    1. Load raw JSON (5-core format)
    2. Convert to interaction datatype
    3. Re-apply 5-core filter (sanity check after implicit conversion)
    4. Encode user/item IDs to consecutive integers
    5. Leave-one-out train/val/test split
    6. Save processed files

Usage:
    # Download data first:
    # wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Musical_Instruments.json.gz
    
"""

import argparse
import gzip
import json
import os
import pickle
from collections import defaultdict
import pandas as pd

# ─────────────────────────────────────────────
# 1. Load raw JSON
# ─────────────────────────────────────────────

def load_raw_json(filepath: str) -> pd.DataFrame:
    """
    Load the Amazon 5-core JSON (one review object per line, gzip-compressed).
    Only keep the three columns we need: reviewerID, asin, unixReviewTime.
    Rating value is intentionally dropped — we treat every interaction as
    implicit positive feedback (the user interacted with the item).
    """
    records = []
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append({
                "user_id":   obj["reviewerID"],
                "item_id":   obj["asin"],
                "rating":    float(obj.get("overall", 0)),
                "timestamp": int(obj.get("unixReviewTime", 0)),
            })

    df = pd.DataFrame(records)  #turn the dict into dataFrame
    print(f"[1] Raw records loaded       : {len(df):,}")
    print(f"    Unique users             : {df['user_id'].nunique():,}")
    print(f"    Unique items             : {df['item_id'].nunique():,}")
    print(f"    Rating distribution:\n{df['rating'].value_counts().sort_index()}\n")
    return df


# ─────────────────────────────────────────────
# 2. Convert to interaction datatype
# ─────────────────────────────────────────────

def to_interacted(df: pd.DataFrame, min_rating: float = 1.0) -> pd.DataFrame:
    """
    Treat every review with rating as an interaction.

    - We keep ALL ratings (min_rating=1.0) because even a 1-star review
      means the user purchased and interacted with the item — which is the
      signal we care about for recommendation.
    - After filtering, drop the rating column and deduplicate
        (a user may have reviewed the same item twice).
    """
    before = len(df)
    df = df[df["rating"] >= min_rating].copy()
    df = df.drop(columns=["rating"])
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    print(f"[2] After interacted conversion (min_rating={min_rating})")
    print(f"    Interactions kept        : {len(df):,}  (dropped {before - len(df):,})\n")
    return df

# ─────────────────────────────────────────────
# 3. K-core filtering
# ─────────────────────────────────────────────

def kcore_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Iteratively remove users and items with fewer than k interactions.
    Even though the raw file is already 5-core, after deduplication the
    counts can drop below k, so we re-apply the filter to be safe.

    """
    iteration = 0
    while True:
        before = len(df)
        # Remove items with < k interactions
        item_counts = df["item_id"].value_counts()
        item_k = item_counts[item_counts >= k]
        df = df[df["item_id"].isin(item_k.index)]
        # Remove users with < k interactions
        user_counts = df["user_id"].value_counts()
        user_k = user_counts[user_counts >= k]
        df = df[df["user_id"].isin(user_k.index)]
        iteration += 1
        if len(df) == before:
            break   # no more rows removed

    print(f"[3] After {k}-core filter ({iteration} iterations)")
    print(f"    Interactions             : {len(df):,}")
    print(f"    Unique users             : {df['user_id'].nunique():,}")
    print(f"    Unique items             : {df['item_id'].nunique():,}")
    return df


# ─────────────────────────────────────────────
# 4. ID encoding
# ─────────────────────────────────────────────

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Map raw string IDs to consecutive integers starting from 0.
    Embedding layers in PyTorch require integer indices.

    Returns:
        df          : DataFrame with integer user_idx and item_idx columns
        user2idx    : dict mapping original user_id -> integer index
        item2idx    : dict mapping original item_id -> integer index
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user2idx = {}
    item2idx = {}
    for i, u in enumerate(users):
        user2idx[u] = i
    for i, item in enumerate(items):
        item2idx[item] = i
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"[4] ID encoding complete")
    print(f"    n_users                  : {n_users:,}")
    print(f"    n_items                  : {n_items:,}\n")
    return df, user2idx, item2idx


# ─────────────────────────────────────────────
# 5. Leave-one-out split
# ─────────────────────────────────────────────

def leave_one_out_split(df: pd.DataFrame) -> tuple[
    dict[int, list[int]],   # train
    dict[int, int],          # val  (1 item per user)
    dict[int, int],          # test (1 item per user)
]:
    """
    Leave-one-out protocol for model training:
      - Sort each user's interactions by timestamp (ascending).
      - The LAST interaction  → test  (ground truth to rank)
      - The 2nd-to-last      → val   (used for early stopping)
      - Everything else      → train

    Returns three dicts keyed by user_idx (integer).
    """

    # Sort by user_idx and timestamp
    df_sorted = df.sort_values(["user_idx", "timestamp"])

    train_dict = defaultdict(list)   # user -> list of item indices
    val_dict   = {}                  # user -> single item index
    test_dict  = {}                  # user -> single item index

    for user_idx, group in df_sorted.groupby("user_idx"):
        items = group["item_idx"].tolist()
        test_dict[user_idx]  = items[-1]
        val_dict[user_idx]   = items[-2]
        train_dict[user_idx] = items[:-2]

    total_train = 0
    for v in train_dict.values():
        total_train += len(v)

    print(f"[5] Leave-one-out split")
    print(f"    Valid users (all splits) : {len(train_dict):,}")
    print(f"    Train interactions       : {total_train:,}")
    print(f"    Val  interactions        : {len(val_dict):,}  ")
    print(f"    Test interactions        : {len(test_dict):,} \n")
    return dict(train_dict), val_dict, test_dict


# ─────────────────────────────────────────────
# 6. Save processed data
# ─────────────────────────────────────────────

def save_processed(
    train_dict: dict,
    val_dict:   dict,
    test_dict:  dict,
    user2idx:   dict,
    item2idx:   dict,
    n_users:    int,
    n_items:    int,
    output_dir: str,
) -> None:
    """
    Save all processed artefacts to output_dir.

    Files produced:
      train.pkl   — dict {user_idx: [item_idx, ...]}
      val.pkl     — dict {user_idx: item_idx}
      test.pkl    — dict {user_idx: item_idx}
      user2idx.pkl
      item2idx.pkl
      dataset_stats.json  — metadata for quick reference
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, obj in [
        ("train.pkl",    train_dict),
        ("val.pkl",      val_dict),
        ("test.pkl",     test_dict),
        ("user2idx.pkl", user2idx),
        ("item2idx.pkl", item2idx),
    ]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(obj, f)

    stats = {
        "n_users":            n_users,
        "n_items":            n_items,
        "n_train":            sum(len(v) for v in train_dict.values()),
        "n_val":              len(val_dict),
        "n_test":             len(test_dict),
        "avg_train_per_user": round(
            sum(len(v) for v in train_dict.values()) / len(train_dict), 2
        ),
    }
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[6] Processed data saved to  : {output_dir}")
    print(f"    Files: train.pkl, val.pkl, test.pkl, user2idx.pkl, item2idx.pkl, dataset_stats.json")
    print(f"\n    === Final Dataset Stats ===")
    for k, v in stats.items():
        print(f"    {k:<28}: {v:,}" if isinstance(v, int) else f"    {k:<28}: {v}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amazon 5-core data pipeline")
    parser.add_argument("--input",      type=str, default="Musical_Instruments.json.gz",
                        help="Path to the raw 5-core JSON(.gz) file")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="Directory to save processed files")
    parser.add_argument("--min_rating", type=float, default=1.0,
                        help="Minimum rating to treat as implicit positive (default 1.0 = keep all)")
    parser.add_argument("--kcore",      type=int,   default=5,
                        help="K for k-core filtering (default 5)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Amazon Review Data — Data Pipeline")
    print("=" * 55 + "\n")

    # Run pipeline
    df = load_raw_json(args.input)
    df = to_interacted(df, min_rating=args.min_rating)
    df = kcore_filter(df, k=args.kcore)
    df, user2idx, item2idx  = encode_ids(df)
    train, val, test = leave_one_out_split(df)

    save_processed(
        train_dict = train,
        val_dict   = val,
        test_dict  = test,
        user2idx   = user2idx,
        item2idx   = item2idx,
        n_users    = len(user2idx),
        n_items    = len(item2idx),
        output_dir = args.output_dir,
    )

    # Generate EDA plots (df still has timestamp for temporal distribution)
    from eda_plots import plot_eda
    plot_eda(df, args.output_dir)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
