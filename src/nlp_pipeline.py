"""Text cleaning, TF-IDF features and matching keywords."""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


def tokenize(text):
    return [word for word in clean_text(text).split()
            if len(word) > 1 and word not in ENGLISH_STOP_WORDS]


class NLPPipeline:
    def __init__(self, items):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        texts = items[["title", "syllabus_text", "skills_tags"]].agg(" ".join, axis=1)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.terms = self.vectorizer.get_feature_names_out()

    def transform(self, text):
        """Use the catalog vocabulary for a new query or description."""
        return self.vectorizer.transform([text])

    def similarities(self, vector):
        return cosine_similarity(vector, self.matrix).ravel()

    def matching_keywords(self, vector, item_index):
        shared = self.matrix[item_index].multiply(vector).tocsr()
        ranked = sorted(zip(shared.indices, shared.data),
                        key=lambda pair: (-pair[1], self.terms[pair[0]]))
        return ", ".join(self.terms[index] for index, value in ranked[:4] if value > 0)
