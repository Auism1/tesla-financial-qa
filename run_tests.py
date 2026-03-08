"""
Run 5 complex test questions against the Tesla QA system and print results.
"""
import os
import sys
import json
from pathlib import Path

# Ensure project root is in path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.indexer import load_index
from src.qa_system import answer_question

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "Compare how 2021 10-K and 2023 10-K describe China market risks. What changed?",
        "type": "Cross-document comparison",
    },
    {
        "id": "Q2",
        "question": "What was the total R&D expense across all four quarters of 2022? Compare to FY2021 from the 2021 annual report.",
        "type": "Numerical calculation",
    },
    {
        "id": "Q3",
        "question": "Which quarter from 2021 to 2024 had the lowest automotive gross margin? What does the MD&A say about that quarter?",
        "type": "Text + table join",
    },
    {
        "id": "Q4",
        "question": "Describe Tesla's free cash flow fluctuation quarter by quarter from 2021 to 2023.",
        "type": "Temporal multi-doc",
    },
    {
        "id": "Q5",
        "question": "How did Tesla's risk factor disclosures about supply chain evolve from 2021 Q1 to 2023 Q3? Cite specific quarters.",
        "type": "Multi-doc synthesis",
    },
]

SEPARATOR = "=" * 80


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(SEPARATOR)
    print("Tesla QA System — Test Suite")
    print(SEPARATOR)

    collection, bm25, chunks = load_index(api_key)
    print(f"Index loaded: {len(chunks)} chunks\n")

    results = []
    for test in TEST_QUESTIONS:
        print(f"\n{SEPARATOR}")
        print(f"[{test['id']}] {test['type']}")
        print(f"Q: {test['question']}")
        print("-" * 80)

        result = answer_question(
            question=test["question"],
            collection=collection,
            bm25=bm25,
            chunks=chunks,
            api_key=api_key,
            debug=True,
        )

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES RETRIEVED ({len(result['sources'])}):")
        for i, s in enumerate(result["sources"][:10]):
            print(f"  [{i+1}] {s['period']} | {s['section']} | pg {s['page']} | {s['chunk_type']} | rrf={s['rrf_score']:.4f}")

        print(f"\nQUERY ANALYSIS: {json.dumps(result.get('query_analysis', {}), indent=2)}")
        print(f"SUB-QUERIES: {result.get('sub_queries', [])}")

        results.append({**test, "result": result})

    print(f"\n{SEPARATOR}")
    print("Test run complete.")
    return results


if __name__ == "__main__":
    main()
