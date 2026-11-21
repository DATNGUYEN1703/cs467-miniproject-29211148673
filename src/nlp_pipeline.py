"""English syllabus preprocessing, TF-IDF features and cosine explanations."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Lowercase and replace punctuation with spaces (preserve word boundaries)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()


def tokenize(text: str) -> list[str]:
    """Remove English stop words and one-character tokens after cleaning."""
    return [word for word in clean_text(text).split()
            if len(word) > 1 and word not in ENGLISH_STOP_WORDS]


class NLPPipeline:
    """Fit once on catalog metadata; transform every query in that vocabulary.

    This project intentionally supports English syllabi and English queries.
    Ratings and feedback never enter TF-IDF fitting.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True, norm="l2", min_df=1,
        )
        self.matrix = None
        self.feature_names = None

    def fit(self, items: pd.DataFrame) -> "NLPPipeline":
        texts = items[["title", "syllabus_text", "skills_tags"]].agg(" ".join, axis=1)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, texts: str | Sequence[str]) -> csr_matrix:
        if self.matrix is None:
            raise RuntimeError("Fit the NLP pipeline before transforming queries.")
        return self.vectorizer.transform([texts] if isinstance(texts, str) else texts)

    def similarities(self, vector: csr_matrix) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Fit the NLP pipeline first.")
        return cosine_similarity(vector, self.matrix).ravel()

    def query_similarities(self, query: str) -> np.ndarray:
        return self.similarities(self.transform(query))

    def top_terms(self, item_index: int, limit: int = 5) -> list[str]:
        row = self.matrix.getrow(item_index)
        ranked = sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], self.feature_names[pair[0]]))
        return [str(self.feature_names[index]) for index, _ in ranked[:limit]]

    def matching_keywords(self, vector: csr_matrix, item_index: int, limit: int = 4) -> list[str]:
        """Shared terms ordered by their contribution to the vector dot product."""
        shared = self.matrix.getrow(item_index).multiply(vector).tocsr()
        ranked = sorted(zip(shared.indices, shared.data), key=lambda pair: (-pair[1], self.feature_names[pair[0]]))
        return [str(self.feature_names[index]) for index, value in ranked[:limit] if value > 0]
