"""Behavioral checks for data, cold-start, filtering and evaluation isolation.

Run from the project root: python -m unittest discover -s tests -v
"""

import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src.evaluate import evaluate, holdout_split
from src.main import CAREER_QUERY, COLD_QUERY, run_required_cases
from src.nlp_pipeline import clean_text, tokenize
from src.recommender import ROOT, CourseRecommender, load_data, validate_data


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.ratings = load_data()
        cls.model = CourseRecommender(cls.items, cls.ratings)

    def test_dataset_schema_size_sparsity_and_ai_student(self):
        self.assertEqual(len(self.items), 36)
        self.assertEqual(self.ratings["user_id"].nunique(), 14)
        self.assertEqual(len(self.ratings), 252)
        self.assertTrue(self.ratings.groupby("user_id").size().ge(6).all())
        missing = 1 - len(self.ratings) / (36 * 14)
        self.assertTrue(0.40 <= missing <= 0.65)
        ai = self.ratings[(self.ratings["user_id"] == "U001") & self.ratings["item_id"].isin([f"C{i:03d}" for i in range(7, 13)])]
        self.assertEqual(len(ai), 6)
        self.assertTrue(ai["rating"].eq(5).all())

    def test_invalid_and_duplicate_ratings_rejected(self):
        with self.assertRaises(ValueError):
            validate_data(self.items, pd.concat([self.ratings, self.ratings.iloc[[0]]]))
        for invalid in [0, 6, 2.5, np.nan]:
            broken = self.ratings.copy()
            broken["rating"] = broken["rating"].astype(float)
            broken.loc[0, "rating"] = invalid
            with self.assertRaises(ValueError):
                validate_data(self.items, broken)

    def test_preprocessing_and_query_vocabulary_reuse(self):
        self.assertEqual(clean_text("  Backend/API, CLOUD! "), "backend api cloud")
        self.assertEqual(tokenize("I want to learn APIs and the cloud"), ["want", "learn", "apis", "cloud"])
        vocabulary = self.model.nlp.vectorizer.vocabulary_.copy()
        vector = self.model.nlp.transform(CAREER_QUERY)
        self.assertEqual(vector.shape[1], self.model.nlp.matrix.shape[1])
        self.assertEqual(vocabulary, self.model.nlp.vectorizer.vocabulary_)
        self.assertIn("cloud deployment", vocabulary)
        self.assertTrue(np.isfinite(self.model.nlp.query_similarities(CAREER_QUERY)).all())

    def test_every_method_filters_seen_courses(self):
        seen = set(self.model.history("U001")["item_id"])
        for method in ["popularity", "content", "cf", "hybrid"]:
            with self.subTest(method=method):
                result = self.model.recommend(method, user_id="U001")
                self.assertEqual(len(result), 5)
                self.assertFalse(set(result["item_id"]) & seen)
                self.assertTrue(np.isfinite(result["score"]).all())
                self.assertTrue(result["score"].is_monotonic_decreasing)

    def test_popularity_minimum_count_and_item_self_exclusion(self):
        popular = self.model.recommend("popularity", top_k=36, min_count=3)
        self.assertTrue(popular["rating_count"].ge(3).all())
        self.assertNotIn("C036", set(popular["item_id"]))
        similar = self.model.recommend("content", item_id="C001")
        self.assertNotIn("C001", set(similar["item_id"]))
        self.assertTrue(similar["keywords"].str.len().gt(0).all())

    def test_backend_query_and_dynamic_cold_start(self):
        result = self.model.recommend("hybrid", query=CAREER_QUERY)
        self.assertEqual(len(result), 5)
        self.assertTrue(result["department"].eq("Software Engineering").all())
        self.assertFalse(self.ratings["item_id"].eq("C036").any())
        self.assertIn("C036", set(self.model.recommend("hybrid", query=COLD_QUERY)["item_id"]))
        self.assertIn("PASS:", run_required_cases(self.items, self.ratings))

    def test_cf_predictions_use_peer_ratings_and_fallback(self):
        # Two positively correlated users. V rates the unseen course 5/5;
        # perturbing only that rating must change U's prediction.
        rows = [("U", "C001", 1), ("U", "C002", 3), ("U", "C003", 5),
                ("V", "C001", 1), ("V", "C002", 3), ("V", "C003", 5), ("V", "C004", 5)]
        tiny = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
        tiny["feedback_comment"] = "Synthetic test record"
        model = CourseRecommender(self.items, tiny)
        self.assertEqual(model.neighbors("U")[0][0], "V")
        first = model.predict_rating("U", "C004")
        tiny.loc[tiny["item_id"] == "C004", "rating"] = 1
        second = CourseRecommender(self.items, tiny).predict_rating("U", "C004")
        self.assertGreater(first, second)
        self.assertTrue(1 <= model.predict_rating("U", "C036") <= 5)

    def test_holdout_disjoint_reproducible_and_label_independent(self):
        train, test = holdout_split(self.ratings)
        pairs = lambda frame: set(zip(frame["user_id"], frame["item_id"]))
        self.assertFalse(pairs(train) & pairs(test))
        self.assertEqual(len(train) + len(test), len(self.ratings))
        pd.testing.assert_frame_equal(train, holdout_split(self.ratings)[0])
        changed = self.ratings.copy()
        mask = [(user, item) in pairs(test) for user, item in zip(changed["user_id"], changed["item_id"])]
        changed.loc[mask, "rating"] = 1
        pd.testing.assert_frame_equal(train, holdout_split(changed)[0])
        altered_model = CourseRecommender(self.items, holdout_split(changed)[0])
        original_model = CourseRecommender(self.items, train)
        pd.testing.assert_frame_equal(altered_model.recommend("hybrid", user_id="U001"),
                                      original_model.recommend("hybrid", user_id="U001"))

    def test_evaluation_metric_denominator_and_saved_recommendations(self):
        metrics, metadata, per_user = evaluate(self.items, self.ratings, output_dir=None)
        self.assertEqual(metadata["test_ratings"], 56)
        self.assertEqual(set(metrics["method"]), {"popularity", "content", "cf", "hybrid"})
        train, test = holdout_split(self.ratings)
        for row in per_user.itertuples():
            ids = set(row.recommended_ids.split())
            seen = set(train.loc[train["user_id"] == row.user_id, "item_id"])
            relevant = set(test.loc[(test["user_id"] == row.user_id) & (test["rating"] >= 4), "item_id"])
            self.assertFalse(ids & seen)
            self.assertEqual(row.precision_at_5, len(ids & relevant) / 5)

    def test_actionable_errors_and_new_student_query(self):
        for kwargs in [{"query": ""}, {"query": "zzzzqwerty"}, {"user_id": "UNKNOWN"}, {"query": CAREER_QUERY, "top_k": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.recommend("hybrid", **kwargs)
        self.assertEqual(len(self.model.recommend("hybrid", query=CAREER_QUERY)), 5)

    def test_cli_direct_script_and_invalid_user(self):
        good = subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), "--query", CAREER_QUERY],
                              cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("5.", good.stdout)
        bad = subprocess.run([sys.executable, "-m", "src.main", "--user", "UNKNOWN"],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("Unknown student", bad.stderr)


if __name__ == "__main__":
    unittest.main()
