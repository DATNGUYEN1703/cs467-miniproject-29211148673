"""Run the three required cases or enter a student ID, query or course ID."""

import argparse

import pandas as pd

from recommender import ROOT, CourseRecommender, load_data

CAREER_QUERY = "I want to learn backend APIs, cloud deployment, and databases"
COLD_QUERY = "serverless backend APIs, cloud functions, cloud deployment and managed databases"


def format_recommendations(frame):
    lines = []
    for rank, row in enumerate(frame.itertuples(), 1):
        keywords = row.keywords or "No shared terms"
        lines.append(f"{rank}. {row.item_id} | {row.title}\n"
                     f"   Score: {row.score:.4f}; cosine: {row.similarity:.4f}; ratings: {row.rating_count}\n"
                     f"   Keywords: {keywords}")
    return "\n".join(lines)


def run_demo(items, ratings):
    new_course = items[items.item_id == "C036"]
    original = items[items.item_id != "C036"]
    model = CourseRecommender(original, ratings)
    ai_history = ratings[(ratings.user_id == "U001") & ratings.item_id.isin([f"C{i:03d}" for i in range(7, 13)])]
    assert len(ai_history) == 6 and ai_history.rating.eq(5).all()
    lines = ["TEST 1 - Career goal query (Hybrid)", f"Query: {CAREER_QUERY}",
             format_recommendations(model.recommend(query=CAREER_QUERY)),
             "\nTEST 2 - Existing student (Content)",
             "U001 rated six AI/data courses C007-C012 5/5. Recommend unseen electives:",
             format_recommendations(model.recommend("content", user_id="U001"))]
    assert len(new_course) == 1 and not ratings.item_id.eq("C036").any()
    expanded = pd.concat([original, new_course], ignore_index=True)
    updated_model = CourseRecommender(expanded, ratings)
    cold_result = updated_model.recommend(query=COLD_QUERY)
    assert "C036" in set(cold_result.item_id)
    lines.extend(["\nTEST 3 - Newly launched course (Hybrid)",
                  f"Added C036: catalog {len(original)} -> {len(expanded)} courses; no new ratings.",
                  f"Query: {COLD_QUERY}", format_recommendations(cold_result),
                  "PASS: the new course appears in Top-5 with zero ratings."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--user", help="Existing student, e.g. U001")
    source.add_argument("--query", help="English career goal")
    source.add_argument("--item", help="Source course, e.g. C001")
    parser.add_argument("--method", choices=["popularity", "content", "cf", "hybrid"])
    parser.add_argument("--demo", action="store_true", help="Run and save all three required cases")
    args = parser.parse_args()
    has_source = any(value is not None for value in [args.user, args.query, args.item])
    if args.demo and (has_source or args.method):
        parser.error("Use --demo on its own.")
    if args.method and not has_source:
        parser.error("Choose a student ID, query or source course.")
    items, ratings = load_data()
    try:
        if not has_source:
            output = run_demo(items, ratings)
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results/sample_outputs.txt").write_text(output, encoding="utf-8")
        else:
            model = CourseRecommender(items, ratings)
            method = args.method or ("content" if args.item is not None else "hybrid")
            output = format_recommendations(model.recommend(method, user_id=args.user,
                                                            query=args.query, item_id=args.item))
        print(output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
