import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def mark_positive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_positive"] = (
        df["event_type"].isin(["like", "favorite"])
        | ((df["event_type"] == "watch_time") & (df["watch_time"] > 60))
    )
    return df


def _watch_weight(timespent: pd.Series) -> pd.Series:
    return pd.cut(
        timespent,
        bins=[60, 100, 200, 300, np.inf],
        labels=[1, 2, 3, 4],
        right=True,
    ).astype(float)


def add_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weight = pd.Series(np.nan, index=df.index)
    weight[df["event_type"] == "favorite"] = 6.0
    weight[df["event_type"] == "like"] = 5.0
    is_wt = (df["event_type"] == "watch_time") & (df["watch_time"] > 60)
    weight[is_wt] = _watch_weight(df.loc[is_wt, "watch_time"])
    df["weight"] = weight
    return df


def build_mappings(df: pd.DataFrame) -> dict[str, dict[int, int]]:
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    return {
        "user2idx": {u: i for i, u in enumerate(user_ids)},
        "idx2user": {i: u for i, u in enumerate(user_ids)},
        "item2idx": {it: i for i, it in enumerate(item_ids)},
        "idx2item": {i: it for i, it in enumerate(item_ids)},
    }


def build_matrix(df: pd.DataFrame, mappings: dict[str, dict[int, int]], binary: bool = False) -> sp.csr_matrix:
    pos = df[df["is_positive"]].copy()
    pos["user_idx"] = pos["user_id"].map(mappings["user2idx"])
    pos["item_idx"] = pos["item_id"].map(mappings["item2idx"])
    pos = pos.dropna(subset=["user_idx", "item_idx"])

    agg = pos.groupby(["user_idx", "item_idx"])["weight"].sum().reset_index()

    n_users = len(mappings["user2idx"])
    n_items = len(mappings["item2idx"])
    data = np.ones(len(agg)) if binary else agg["weight"].values
    return sp.csr_matrix(
        (data, (agg["user_idx"].astype(int), agg["item_idx"].astype(int))),
        shape=(n_users, n_items),
    )


def get_day_boundaries(df: pd.DataFrame, time_col: str) -> pd.Series:
    ts = df[time_col]
    shifted = ts - pd.Timedelta(hours=9)
    date = shifted.dt.date
    sorted_dates = sorted(date.unique())
    date_to_day = {d: i + 1 for i, d in enumerate(sorted_dates)}
    return date.map(date_to_day)


def precision_at_k(recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int = 20) -> float:
    scores = []
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        if not gt:
            continue
        hits = len(set(recs[:k]) & gt)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def build_ground_truth(df: pd.DataFrame, label_day: int, user2idx: dict[int, int]) -> dict[int, set[int]]:
    gt = (df[(df["day"] == label_day) & df["is_positive"]]
          .groupby("user_id")["item_id"].apply(set).to_dict())
    return {u: items for u, items in gt.items() if u in user2idx}


def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(
        path, engine="pyarrow", compression="zstd", compression_level=9, index=False
    )