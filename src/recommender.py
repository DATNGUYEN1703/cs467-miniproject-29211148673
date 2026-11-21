"""Popularity, TF-IDF content, user-based CF and hybrid elective advisor."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.nlp_pipeline import NLPPipeline

ROOT = Path(__file__).resolve().parents[1]
ITEM_COLUMNS = {"item_id", "course_code", "title", "department", "syllabus_text", "skills_tags"}
RATING_COLUMNS = {"user_id", "item_id", "rating", "feedback_comment"}


def validate_data(items: pd.DataFrame, ratings: pd.DataFrame) -> None:
    if not ITEM_COLUMNS.issubset(items.columns) or not RATING_COLUMNS.issubset(ratings.columns):
        raise ValueError("CSV columns do not match the required items/ratings schemas.")
    if items.empty or ratings.empty:
        raise ValueError("The catalog and ratings must not be empty.")
    for frame, columns in [(items, ITEM_COLUMNS), (ratings, RATING_COLUMNS - {"rating"})]:
        for column in sorted(columns):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing/empty values in {column}.")
    if items["item_id"].duplicated().any() or items["course_code"].duplicated().any():
        raise ValueError("Duplicate course IDs/codes.")
    if ratings.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate user-course ratings.")
    if not set(ratings["item_id"]).issubset(set(items["item_id"])):
        raise ValueError("A rating refers to an unknown course.")
    numeric = pd.to_numeric(ratings["rating"], errors="coerce")
    if numeric.isna().any() or not numeric.between(1, 5).all() or not numeric.eq(numeric.round()).all():
        raise ValueError("Ratings must be integers from 1 to 5.")


def load_data(data_dir: Path = ROOT / "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_csv(data_dir / "items.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    validate_data(items, ratings)
    return items, ratings


class CourseRecommender:
    """All rating-derived state is computed only from the supplied ratings.

    Supply training ratings for evaluation, and full ratings for the live demo.
    The full catalog is available at serving time, including unrated courses.
    """

    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame):
        validate_data(items, ratings)
        self.items = items.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.ratings["rating"] = pd.to_numeric(self.ratings["rating"])
        self.positions = {item_id: index for index, item_id in enumerate(self.items["item_id"])}
        self.nlp = NLPPipeline().fit(self.items)
        self.global_mean = float(self.ratings["rating"].mean())
        stats = self.ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        self.stats = stats.reindex(self.items["item_id"]).copy()
        self.stats["count"] = self.stats["count"].fillna(0).astype(int)
        self.stats["mean"] = self.stats["mean"].fillna(self.global_mean)
        # A prior of three global-mean ratings reduces small-sample volatility.
        self.stats["weighted"] = ((self.stats["count"] * self.stats["mean"] + 3 * self.global_mean)
                                  / (self.stats["count"] + 3))
        self.matrix = self.ratings.pivot(index="user_id", columns="item_id", values="rating")
        self.user_means = self.matrix.mean(axis=1)

    def history(self, user_id: str) -> pd.DataFrame:
        if user_id not in self.matrix.index:
            raise ValueError(f"Unknown student {user_id}. Use a free-text query for a new student.")
        return self.ratings[self.ratings["user_id"] == user_id]

    def profile(self, user_id: str) -> csr_matrix:
        history = self.history(user_id)
        indices = [self.positions[item] for item in history["item_id"]]
        weights = np.maximum(history["rating"].to_numpy(dtype=float) - 3, 0)
        if weights.sum() == 0:
            weights = history["rating"].to_numpy(dtype=float)
        profile = csr_matrix(weights.reshape(1, -1)) @ self.nlp.matrix[indices]
        return normalize(profile)

    def neighbors(self, user_id: str, limit: int = 5) -> list[tuple[str, float]]:
        """Positive Pearson similarity on >=3 co-ratings, shrunk by n/(n+5)."""
        self.history(user_id)
        target = self.matrix.loc[user_id]
        neighbors = []
        for other in self.matrix.index:
            if other == user_id:
                continue
            peer = self.matrix.loc[other]
            common = target.notna() & peer.notna()
            n_common = int(common.sum())
            if n_common < 3:
                continue
            left = target[common].to_numpy(dtype=float)
            right = peer[common].to_numpy(dtype=float)
            left, right = left - left.mean(), right - right.mean()
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            similarity = float(left @ right / denominator) if denominator > 0 else 0.0
            similarity *= n_common / (n_common + 5)
            if similarity > 0:
                neighbors.append((str(other), similarity))
        return sorted(neighbors, key=lambda pair: (-pair[1], pair[0]))[:limit]

    def _predict(self, user_id: str, item_id: str, neighbors: list[tuple[str, float]]) -> float:
        numerator, denominator = 0.0, 0.0
        for other, similarity in neighbors:
            value = self.matrix.loc[other].get(item_id, np.nan)
            if pd.notna(value):
                numerator += similarity * (float(value) - self.user_means[other])
                denominator += abs(similarity)
        prediction = (self.user_means[user_id] + numerator / denominator if denominator > 0
                      else float(self.stats.loc[item_id, "weighted"]))
        return float(np.clip(prediction, 1, 5))

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if item_id not in self.positions:
            raise ValueError(f"Unknown course {item_id}.")
        return self._predict(user_id, item_id, self.neighbors(user_id))

    def recommend(self, method: str = "hybrid", user_id: str | None = None,
                  query: str | None = None, item_id: str | None = None,
                  top_k: int = 5, min_count: int = 2) -> pd.DataFrame:
        """Return unseen courses with score components and shared keywords.

        Content mode accepts exactly one of a user, query or source item.
        Hybrid accepts a user, a query, or both (the explicit goal takes priority).
        A zero-vector query is rejected so unrelated items are not labeled matches.
        """
        methods = {"popularity", "content", "cf", "hybrid"}
        if method not in methods:
            raise ValueError(f"Choose a method from {sorted(methods)}.")
        if top_k < 1 or min_count < 1:
            raise ValueError("top_k and min_count must be positive integers.")
        if method == "content" and sum(value is not None for value in [user_id, query, item_id]) != 1:
            raise ValueError("Content mode requires exactly one user, query or source item.")
        if item_id is not None and method != "content":
            raise ValueError("Source-item recommendations use method='content'.")
        if method == "cf" and (user_id is None or query is not None):
            raise ValueError("Collaborative filtering requires an existing user and no query.")
        if method == "popularity" and query is not None:
            raise ValueError("Popularity does not use a text query.")
        if method == "hybrid" and user_id is None and query is None:
            raise ValueError("Hybrid mode needs a user ID or a career-goal query.")

        seen = set(self.history(user_id)["item_id"]) if user_id is not None else set()
        vector = None
        if query is not None:
            if not query.strip():
                raise ValueError("The goal statement must not be empty.")
            vector = self.nlp.transform(query)
            if vector.nnz == 0:
                raise ValueError("No query terms match the English catalog. Try backend APIs, cloud, data or security.")
        elif item_id is not None:
            if item_id not in self.positions:
                raise ValueError(f"Unknown course {item_id}.")
            vector = self.nlp.matrix[self.positions[item_id]]
            seen.add(item_id)
        elif user_id is not None and method in {"content", "hybrid"}:
            vector = self.profile(user_id)

        result = self.items[["item_id", "course_code", "title", "department"]].copy()
        result["rating_count"] = self.stats["count"].to_numpy()
        result["average_rating"] = self.stats["mean"].to_numpy()
        result["popularity_rating"] = self.stats["weighted"].to_numpy()
        result["similarity"] = self.nlp.similarities(vector) if vector is not None else 0.0
        result["cf_prediction"] = np.nan
        if user_id is not None and method in {"cf", "hybrid"}:
            neighbors = self.neighbors(user_id)
            result["cf_prediction"] = [self._predict(user_id, course, neighbors) for course in result["item_id"]]
        normalized_popularity = (result["popularity_rating"] - 1) / 4
        if method == "popularity":
            result["score"] = result["popularity_rating"]
            result = result[result["rating_count"] >= min_count]
        elif method == "content":
            result["score"] = result["similarity"]
        elif method == "cf":
            result["score"] = result["cf_prediction"]
        elif user_id is not None:
            result["score"] = (0.55 * result["similarity"] + 0.25 * (result["cf_prediction"] - 1) / 4
                               + 0.20 * normalized_popularity)
        else:
            result["score"] = 0.75 * result["similarity"] + 0.25 * normalized_popularity
        result = (result[~result["item_id"].isin(seen)]
                  .sort_values(["score", "item_id"], ascending=[False, True]).head(top_k).copy())
        result["keywords"] = [", ".join(self.nlp.matching_keywords(vector, self.positions[course]))
                              if vector is not None else "" for course in result["item_id"]]
        result["explanation"] = [
            (f"Shared terms: {keywords}" if keywords else "No shared text terms; rating evidence only")
            + ("; new course, no ratings (global-mean prior)" if count == 0 else f"; {count} ratings")
            for keywords, count in zip(result["keywords"], result["rating_count"])
        ]
        return result.reset_index(drop=True)
