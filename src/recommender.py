"""Popularity, content-based, user-based CF and hybrid recommendations."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from nlp_pipeline import NLPPipeline

ROOT = Path(__file__).resolve().parents[1]


def load_data():
    items = pd.read_csv(ROOT / "data/items.csv")
    ratings = pd.read_csv(ROOT / "data/ratings.csv")
    return items, ratings


class CourseRecommender:
    def __init__(self, items, ratings):
        self.items = items.set_index("item_id").copy()
        self.nlp = NLPPipeline(self.items)
        stats = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        self.average = stats["mean"].reindex(self.items.index).fillna(ratings.rating.mean())
        self.count = stats["count"].reindex(self.items.index, fill_value=0)
        self.matrix = ratings.pivot(index="user_id", columns="item_id", values="rating")
        self.matrix = self.matrix.reindex(columns=self.items.index)
        self.user_means = self.matrix.mean(axis=1)
        # Pearson correlation uses shared ratings; at least three are required.
        self.user_similarity = self.matrix.T.corr(min_periods=3)

    def history(self, user_id):
        if user_id not in self.matrix.index:
            raise ValueError(f"Unknown student {user_id}. Use --query for a new student.")
        return self.matrix.loc[user_id].dropna()

    def profile(self, user_id):
        history = self.history(user_id)
        weights = np.maximum(history.to_numpy() - 3, 0)
        if weights.sum() == 0:
            weights = history.to_numpy()
        indices = self.items.index.get_indexer(history.index)
        vector = csr_matrix(weights.reshape(1, -1)) @ self.nlp.matrix[indices]
        return normalize(vector)

    def predict_ratings(self, user_id):
        """Predict ratings using up to five positively correlated students."""
        self.history(user_id)
        neighbors = self.user_similarity[user_id].drop(user_id).dropna()
        neighbors = neighbors[neighbors > 0].nlargest(5)
        peers = self.matrix.loc[neighbors.index]
        deviations = peers.sub(self.user_means.loc[neighbors.index], axis=0)
        numerator = deviations.mul(neighbors, axis=0).sum()
        denominator = peers.notna().mul(neighbors, axis=0).sum()
        predictions = self.user_means[user_id] + numerator / denominator.replace(0, np.nan)
        return predictions.fillna(self.average).clip(1, 5)

    def recommend(self, method="hybrid", user_id=None, query=None, item_id=None):
        if method not in {"popularity", "content", "cf", "hybrid"}:
            raise ValueError("Unknown recommendation method.")
        if sum(value is not None for value in [user_id, query, item_id]) != 1:
            raise ValueError("Provide exactly one student ID, query or source course.")
        if method in {"popularity", "cf"} and user_id is None:
            raise ValueError(f"{method} requires an existing student ID.")
        if item_id is not None and method != "content":
            raise ValueError("Use content for item-to-item recommendations.")

        seen = set(self.history(user_id).index) if user_id is not None else set()
        if query is not None:
            vector = self.nlp.transform(query)
            if vector.nnz == 0:
                raise ValueError("No matching words. Try an English goal about cloud, data or security.")
        elif item_id is not None:
            if item_id not in self.items.index:
                raise ValueError(f"Unknown course {item_id}.")
            vector = self.nlp.matrix[self.items.index.get_loc(item_id)]
            seen.add(item_id)
        else:
            vector = self.profile(user_id)

        result = self.items[["course_code", "title"]].copy()
        result["similarity"] = self.nlp.similarities(vector)
        result["rating_count"] = self.count
        normalized_average = (self.average - 1) / 4
        if method == "popularity":
            result["score"] = self.average
            result = result[result.rating_count >= 2]
        elif method == "content":
            result["score"] = result.similarity
        elif method == "cf":
            result["score"] = self.predict_ratings(user_id)
        elif user_id is not None:
            result["score"] = (0.70 * result.similarity
                               + 0.20 * (self.predict_ratings(user_id) - 1) / 4
                               + 0.10 * normalized_average)
        else:
            result["score"] = 0.75 * result.similarity + 0.25 * normalized_average
        result = result.drop(index=list(seen), errors="ignore")
        result = result.sort_values(["score", "item_id"], ascending=[False, True]).head(5)
        result["keywords"] = [self.nlp.matching_keywords(vector, self.items.index.get_loc(course))
                              for course in result.index]
        return result.reset_index()
