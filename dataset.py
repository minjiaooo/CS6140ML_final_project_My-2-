"""
dataset.py — Dataset and DataLoader for Recommendation Models

Shared by both MF and Two-Tower models.

Core Logic:
    Converts the {user: [item1, item2, ...]} mapping from train.pkl into
    training triplets (user, pos_item, neg_item).

    pos_item = items the user actually interacted with (positive samples)
    neg_item = randomly sampled items the user has not interacted with (negative samples)
"""

import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────
# 1. Training Dataset (with negative sampling)
# ──────────────────────────────────────────

class TrainDataset(Dataset):
    """ 
    Training Set: Each sample is a triplet (user, pos_item, neg_item)

    Parameters:
        train_dict : dict, {user_idx: [item_idx, ...]}. Loaded from train.pkl
        self.samples: list, [(user, pos_item), ...]
        self.user_item_set: dict, {user_idx: {item_idx, ...}}
        n_items : int, total number of items, used for random negative sampling
        n_neg : int, number of negative samples per positive sample 
    
    """

    def __init__(self, train_dict: dict, n_items: int, n_neg: int = 1):
        self.n_items = n_items
        self.n_neg = n_neg

        # Flatten train_dict into [(user, pos_item), ...] for indexing
        # Store each user's items as a set for fast negative sampling lookup
        self.samples = []
        self.user_item_set = {}
        for user, items in train_dict.items():
            for item in items:
                self.samples.append((user, item))
            self.user_item_set[user] = set(items)

        print(f"Total training samples (positive): {len(self.samples):,}")

    def __len__(self):
        # DataLoader needs to know the total number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one training sample: (user, pos_item, neg_item)
        """
        user, pos_item = self.samples[idx]

        # Negative sampling: randomly pick an item the user has not interacted with
        # Set a max retry limit to avoid infinite loops when users have many interactions
        max_tries = self.n_items * 2
        neg_item = random.randint(0, self.n_items - 1)
        for _ in range(max_tries):
            if neg_item not in self.user_item_set[user]:
                break
            neg_item = random.randint(0, self.n_items - 1)

        # Return tensors, as PyTorch models only accept tensor types
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


# ──────────────────────────────────────────
# 2. Evaluation Dataset (val / test)
# ──────────────────────────────────────────

class EvalDataset(Dataset):
    """
    Evaluation set: each sample is (user, pos_item, neg_items)

    Evaluation protocol: 99 negatives strategy
        For each user, use 1 ground truth item + 99 random negative samples.
        The model ranks among these 100 items to see where the ground truth lands.
    Parameters:
        eval_dict: dict, {user_idx: item_idx}; loaded from val.pkl or test.pkl
        train_dict: dict,{user_idx: [item_idx, ...]}; used to exclude interacted items
        self.user_item_set: dict, {user_idx: {item_idx, ...}}
        n_items: int, total number of items
        n_neg: int, number of negative samples (default: 99)
    """

    def __init__(
        self,
        eval_dict: dict,
        train_dict: dict,
        n_items: int,
        n_neg: int = 99,
        exclude_dict: dict = None,
    ):
        self.n_items = n_items
        self.n_neg = n_neg
        self.samples = list(eval_dict.items())  # [(user, pos_item), ...]

        # All items each user has interacted with, to be excluded during negative sampling
        self.user_item_set = {}
        for user, items in train_dict.items():
            self.user_item_set[user] = set(items)
        # Add eval positive samples
        for user, item in eval_dict.items():
            self.user_item_set[user].add(item)
        # Add extra exclusions (e.g., val items when building test set)
        if exclude_dict is not None:
            for user, item in exclude_dict.items():
                self.user_item_set[user].add(item)

        # Precompute negative samples 
        self.neg_samples = {}
        for user, pos_item in self.samples:
            negs = []
            while len(negs) < n_neg:
                neg = random.randint(0, n_items - 1)
                if neg not in self.user_item_set[user]:
                    negs.append(neg)
            self.neg_samples[user] = negs

        print(f"Total eval samples: {len(self.samples):,}  (1 positive + {n_neg} negatives per user)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one evaluation sample:
            user       : tensor (1,)
            pos_item   : tensor (1,)
            neg_items  : tensor (n_neg,)
        """
        user, pos_item = self.samples[idx]
        neg_items = self.neg_samples[user]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )


# ──────────────────────────────────────────
# 3. DataLoader Factory Function
# ──────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    batch_size: int = 1024,
    n_neg_train: int = 1,
    n_neg_eval: int = 99,
):
    """
    Load processed pkl files and return three DataLoaders.

    Parameters:
        data_dir : str, path to the processed data folder (containing train/val/test.pkl)
        batch_size : int, number of samples per batch (for training)
        n_neg_train : int, number of negative samples per positive sample during training
        n_neg_eval : int, number of negative samples per user during evaluation (standard: 99)

    Returns:
        train_loader, val_loader, test_loader, n_users, n_items
    """
    import os

    # Load pkl files
    def load(name):
        with open(os.path.join(data_dir, name), "rb") as f:
            return pickle.load(f)

    train_dict = load("train.pkl")
    val_dict = load("val.pkl")
    test_dict = load("test.pkl")
    user2idx = load("user2idx.pkl")
    item2idx = load("item2idx.pkl")

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"\nDataset info: {n_users:,} users, {n_items:,} items")

    # Build Datasets
    train_dataset = TrainDataset(train_dict, n_items, n_neg=n_neg_train)
    val_dataset = EvalDataset(val_dict,  train_dict, n_items, n_neg=n_neg_eval)
    test_dataset  = EvalDataset(test_dict, train_dict, n_items, n_neg=n_neg_eval, exclude_dict=val_dict)

    # Build DataLoaders
    # shuffle=True: shuffle training data each epoch 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, n_users, n_items
