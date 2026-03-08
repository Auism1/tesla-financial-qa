"""
Document ingestion pipeline.
Parses all Markdown files (Tesla SEC 10-K and 10-Q filings) and builds
the vector + BM25 index.

Usage:
    python ingest.py              # full pipeline
    python ingest.py --index-only # skip parsing, rebuild index from existing chunks.json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = Path(__file__).parent / "dataset"
CHUNKS_PATH = Path(__file__).parent / "data" / "chunks.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-only", action="store_true",
                        help="Skip parsing, rebuild index from existing chunks.json")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 60)
    print("Tesla Financial Reports Ingestion Pipeline")
    print("=" * 60)

    if args.index_only:
        if not CHUNKS_PATH.exists():
            print(f"ERROR: {CHUNKS_PATH} not found. Run without --index-only first.")
            sys.exit(1)
        print(f"\n[Skipping parse] Loading existing {CHUNKS_PATH}...")
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
    else:
        from src.json_parser import parse_all_markdowns
        from src.chunker import merge_and_rechunk

        md_files = list(DATASET_DIR.rglob("*.md"))
        print(f"\n[1/3] Parsing {len(md_files)} Markdown files...")
        raw_chunks = parse_all_markdowns(DATASET_DIR)
        print(f"Total raw chunks: {len(raw_chunks)}")

        print("\n[2/3] Semantic chunking...")
        chunks = merge_and_rechunk(raw_chunks)
        print(f"Total final chunks: {len(chunks)}")

        years = sorted(set(c.get("year") for c in chunks))
        periods = sorted(set(c.get("period") for c in chunks))
        tables = sum(1 for c in chunks if c.get("chunk_type") == "table")
        texts  = sum(1 for c in chunks if c.get("chunk_type") == "text")
        quarterly = sum(1 for c in chunks if c.get("doc_type") == "quarterly_report")
        annual    = sum(1 for c in chunks if c.get("doc_type") == "annual_report")
        print(f"  Years: {years}")
        print(f"  Periods: {len(periods)}")
        print(f"  Table chunks: {tables} | Text chunks: {texts}")
        print(f"  Quarterly report chunks: {quarterly} | Annual report chunks: {annual}")

        CHUNKS_PATH.parent.mkdir(exist_ok=True)
        with open(CHUNKS_PATH, "w") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n[3/3] Building vector + BM25 index...")
    from src.indexer import build_index
    build_index(chunks, api_key)

    print("\n✓ Ingestion complete!")


if __name__ == "__main__":
    main()
