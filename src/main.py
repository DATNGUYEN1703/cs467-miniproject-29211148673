"""CLI demo: python src/main.py --demo; use --help for custom recommendations."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recommender import ROOT, CourseRecommender, load_data

CAREER_QUERY = "I want to learn backend APIs, cloud deployment, and databases"
COLD_QUERY = "I want to build serverless backend APIs with cloud functions, cloud deployment and managed databases"


def format_recommendations(frame: pd.DataFrame) -> str:
    lines = []
    for rank, row in enumerate(frame.itertuples(), 1):
        cf = f" | CF={row.cf_prediction:.3f}/5" if pd.notna(row.cf_prediction) else ""
        lines.append(f"{rank}. {row.item_id} {row.course_code} | {row.title}\n"
                     f"   score={row.score:.4f} | cosine={row.similarity:.4f}{cf}"
                     f" | ratings={row.rating_count}\n   {row.explanation}")
    return "\n".join(lines) if lines else "No eligible unseen courses."


def run_required_cases(items, ratings) -> str:
    model = CourseRecommender(items, ratings)
    lines = ["MP-A03: THREE REQUIRED TEST CASES", "Dataset is fictional; scores are ranking signals, not probabilities.",
             "English syllabi and English goal statements are supported.\n",
             "TEST 1 - Career goal query (hybrid)", f"Query: {CAREER_QUERY}",
             format_recommendations(model.recommend("hybrid", query=CAREER_QUERY)),
             "\nTEST 2 - Existing AI/data student U001 (content and hybrid comparison)",
             "Observed AI/data courses rated 5/5:"]
    ai = ratings[(ratings["user_id"] == "U001") & (ratings["rating"] == 5)].merge(items, on="item_id")
    for row in ai.itertuples():
        lines.append(f"  {row.item_id}: {row.title} ({row.rating}/5)")
    recommendations = model.recommend("hybrid", user_id="U001")
    assert not set(recommendations["item_id"]) & set(model.history("U001")["item_id"])
    lines.extend(["Content-based next unseen electives:",
                  format_recommendations(model.recommend("content", user_id="U001")),
                  "Hybrid next unseen electives:", format_recommendations(recommendations),
                  "\nTEST 3 - Add a newly launched course without ratings"])
    new_course = items[items["item_id"] == "C036"]
    assert len(new_course) == 1 and not ratings["item_id"].eq("C036").any()
    old_catalog = items[items["item_id"] != "C036"].copy()
    before = CourseRecommender(old_catalog, ratings)
    assert "C036" not in set(before.recommend("hybrid", query=COLD_QUERY)["item_id"])
    expanded_catalog = pd.concat([old_catalog, new_course], ignore_index=True)
    after = CourseRecommender(expanded_catalog, ratings)
    cold_recommendations = after.recommend("hybrid", query=COLD_QUERY)
    assert "C036" in set(cold_recommendations["item_id"]), "New course must appear in relevant Top-5."
    lines.extend([f"Catalog before: {len(old_catalog)}; after: {len(expanded_catalog)}; ratings unchanged: {len(ratings)}",
                  f"Added C036: {new_course.iloc[0]['title']} (0 student ratings)",
                  f"Query: {COLD_QUERY}", format_recommendations(cold_recommendations),
                  "PASS: the new unrated elective appears in Top-5 via its syllabus terms."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run all three required test cases (default without arguments).")
    parser.add_argument("--user", help="Existing student ID, e.g. U001.")
    parser.add_argument("--query", help="English free-text career goal.")
    parser.add_argument("--item", help="Source course ID for item-to-item content matching, e.g. C001.")
    parser.add_argument("--method", choices=["popularity", "content", "cf", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-samples", action="store_true", help="Save the three demo cases in results/sample_outputs.txt.")
    args = parser.parse_args()
    if args.demo and any([args.user, args.query is not None, args.item, args.method]):
        parser.error("Use --demo on its own, optionally with --save-samples.")
    default_demo = len(sys.argv) == 1
    if args.save_samples and not (args.demo or default_demo):
        parser.error("--save-samples must be used with --demo.")
    try:
        items, ratings = load_data()
        if args.demo or default_demo:
            output = run_required_cases(items, ratings)
            print(output)
            if args.save_samples:
                destination = ROOT / "results" / "sample_outputs.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(output, encoding="utf-8")
                print(f"Saved to {destination}")
        else:
            model = CourseRecommender(items, ratings)
            method = args.method or ("content" if args.item else "hybrid")
            print(f"Method: {method}; ranking scores are not probabilities.")
            print(format_recommendations(model.recommend(method, user_id=args.user, query=args.query,
                                                         item_id=args.item, top_k=args.top_k)))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
