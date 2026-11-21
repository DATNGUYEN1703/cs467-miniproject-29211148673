"""Reproducible per-user holdout ranking evaluation and saved charts.

Run: python -m src.evaluate   OR   python src/evaluate.py
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.recommender import ROOT, CourseRecommender, load_data

METHODS = ["popularity", "content", "cf", "hybrid"]


def holdout_split(ratings: pd.DataFrame, seed: int = 42, fraction: float = 0.2):
    """Randomly hide ceil(20%) per user, leaving >=4 training ratings.

    No label-based sampling: even a user with zero hidden relevant items stays
    in macro Precision@5. IDs are stably sorted before seeded sampling.
    """
    if not 0 < fraction < 1:
        raise ValueError("Test fraction must be between zero and one.")
    ratings = ratings.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    hidden_indices = []
    for _, group in ratings.groupby("user_id", sort=True):
        n_test = min(max(1, int(np.ceil(len(group) * fraction))), len(group) - 4)
        if n_test < 1:
            raise ValueError("Each evaluation user needs at least five ratings.")
        hidden_indices.extend(rng.choice(group.index.to_numpy(), size=n_test, replace=False))
    test = ratings.loc[sorted(hidden_indices)].copy()
    train = ratings.drop(index=hidden_indices).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(items: pd.DataFrame, ratings: pd.DataFrame, seed: int = 42,
             top_k: int = 5, output_dir: Path | None = ROOT / "results"):
    train, test = holdout_split(ratings, seed=seed)
    model = CourseRecommender(items, train)
    records = []
    for user_id, hidden in test.groupby("user_id", sort=True):
        relevant = set(hidden.loc[hidden["rating"] >= 4, "item_id"])
        seen = set(train.loc[train["user_id"] == user_id, "item_id"])
        for method in METHODS:
            recommended = model.recommend(method, user_id=user_id, top_k=top_k)
            ids = recommended["item_id"].tolist()
            hits = len(set(ids) & relevant)
            records.append({"user_id": user_id, "method": method, "hits": hits,
                            "precision_at_5": hits / top_k,
                            "recall_at_5": hits / len(relevant) if relevant else np.nan,
                            "hidden_relevant": len(relevant), "returned": len(ids),
                            "candidate_count": len(items) - len(seen),
                            "recommended_ids": " ".join(ids)})
    per_user = pd.DataFrame(records)
    metrics = per_user.groupby("method", sort=False).agg(
        precision_at_5=("precision_at_5", "mean"), recall_at_5=("recall_at_5", "mean"),
        users=("user_id", "count"), hits=("hits", "sum"),
    ).reset_index()
    actual = test["rating"].to_numpy(dtype=float)
    predictions = np.array([model.predict_rating(row.user_id, row.item_id) for row in test.itertuples()])
    numeric = {"method": "cf", "mae": float(np.abs(actual - predictions).mean()),
               "rmse": float(np.sqrt(np.square(actual - predictions).mean()))}
    metadata = {"seed": seed, "top_k": top_k, "relevant_rating_min": 4,
                "train_ratings": len(train), "test_ratings": len(test),
                "test_fraction_actual": len(test) / len(ratings),
                "users": int(test["user_id"].nunique()),
                "users_with_hidden_relevant": int(test.loc[test["rating"] >= 4, "user_id"].nunique()),
                "catalog_size": len(items), "rating_prediction_metrics": numeric,
                "protocol": "Per-user seeded random holdout; rank the full catalog excluding training history. Unobserved items count as non-hits. TF-IDF sees catalog metadata only."}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_dir / "metrics.csv", index=False)
        per_user.to_csv(output_dir / "per_user_metrics.csv", index=False)
        train.to_csv(output_dir / "train_ratings.csv", index=False)
        test.to_csv(output_dir / "test_ratings.csv", index=False)
        pd.DataFrame({"user_id": test["user_id"], "item_id": test["item_id"],
                      "actual_rating": actual, "cf_prediction": predictions}).to_csv(
                          output_dir / "rating_predictions.csv", index=False)
        (output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        create_charts(items, ratings, metrics, output_dir)
        summary = ("Offline evaluation (seed=42, relevant >=4, K=5)\n"
                   + metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}")
                   + f"\n\nCF MAE={numeric['mae']:.4f}; RMSE={numeric['rmse']:.4f}\n"
                   + json.dumps(metadata, indent=2) + "\n")
        (output_dir / "evaluation_summary.txt").write_text(summary.replace("seed=42", f"seed={seed}"), encoding="utf-8")
    return metrics, metadata, per_user


def create_charts(items, ratings, metrics, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    counts = ratings["rating"].value_counts().reindex(range(1, 6), fill_value=0)
    axes[0].bar(counts.index.astype(str), counts, color="#246B8E")
    axes[0].set(xlabel="Student rating", ylabel="Number of ratings", title="252 synthetic course ratings")
    for index, value in enumerate(counts):
        axes[0].text(index, value + 0.7, str(value), ha="center")
    departments = items["department"].value_counts().sort_values()
    axes[1].barh(departments.index, departments, color="#2D8C78")
    axes[1].set(xlabel="Number of courses", title="36 elective courses by department")
    axes[1].set_xlim(0, int(departments.max()) + 1)
    fig.savefig(output_dir / "data_distribution.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = ["#9AA6B2", "#267DA5", "#D69432", "#2D8C78"]
    bars = ax.bar(metrics["method"], metrics["precision_at_5"], color=colors, width=0.6)
    ax.set(ylabel="Macro Precision@5", xlabel="Recommendation method",
           title="Held-out recommendation quality (14 simulated students)",
           ylim=(0, max(0.3, float(metrics["precision_at_5"].max()) + 0.10)))
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in metrics["precision_at_5"]], padding=5)
    ax.text(0, -0.22, "Random per-user holdout; relevance >= 4/5. Synthetic data, single split.",
            transform=ax.transAxes, fontsize=9, color="#555555")
    fig.savefig(output_dir / "evaluation_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    items, ratings = load_data()
    metrics, metadata, _ = evaluate(items, ratings, seed=args.seed, output_dir=args.output_dir)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(metadata, indent=2))
    print(f"Saved evaluation tables, split and charts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
