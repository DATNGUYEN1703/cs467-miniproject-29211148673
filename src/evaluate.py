"""Run a reproducible holdout evaluation and save two charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import ROOT, CourseRecommender, load_data


def holdout_split(ratings):
    """Hide 20% (rounded up) of each student's ratings without using labels."""
    ratings = ratings.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(42)
    hidden = []
    for _, group in ratings.groupby("user_id"):
        hidden.extend(rng.choice(group.index, size=int(np.ceil(len(group) * 0.2)), replace=False))
    return ratings.drop(index=hidden), ratings.loc[sorted(hidden)]


def evaluate(items, ratings):
    train, test = holdout_split(ratings)
    model = CourseRecommender(items, train)
    rows = []
    for method in ["popularity", "content", "cf", "hybrid"]:
        precisions = []
        for user_id, hidden in test.groupby("user_id"):
            relevant = set(hidden.loc[hidden.rating >= 4, "item_id"])
            recommended = set(model.recommend(method, user_id=user_id).item_id)
            precisions.append(len(recommended & relevant) / 5)
        rows.append({"method": method, "precision_at_5": np.mean(precisions)})
    predictions = {user: model.predict_ratings(user) for user in test.user_id.unique()}
    errors = np.array([row.rating - predictions[row.user_id][row.item_id]
                       for row in test.itertuples()])
    return pd.DataFrame(rows), np.abs(errors).mean(), np.sqrt((errors ** 2).mean())


def save_charts(items, ratings, metrics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ratings.rating.value_counts().reindex(range(1, 6), fill_value=0).plot.bar(ax=axes[0], color="steelblue", rot=0)
    axes[0].set(title=f"{len(ratings)} course ratings", xlabel="Rating", ylabel="Count")
    items.department.value_counts().sort_values().plot.barh(ax=axes[1], color="seagreen")
    axes[1].set(title=f"{len(items)} elective courses", xlabel="Count", ylabel="Department")
    fig.savefig(output_dir / "data_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(metrics.method, metrics.precision_at_5,
                  color=["gray", "steelblue", "orange", "seagreen"])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(title="Recommendation quality: random holdout, seed 42",
           xlabel="Method", ylabel="Macro Precision@5",
           ylim=(0, max(metrics.precision_at_5) + 0.08))
    fig.savefig(output_dir / "evaluation_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    items, ratings = load_data()
    metrics, mae, rmse = evaluate(items, ratings)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    save_charts(items, ratings, metrics, output_dir)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"CF MAE: {mae:.4f}; RMSE: {rmse:.4f}")
