"""
文档摄取流程。
解析所有 MinerU JSON 文件（特斯拉 SEC 10-K 和 10-Q 财报）并构建
向量 + BM25 索引。

用法：
    python ingest.py              # 完整流程
    python ingest.py --index-only # 跳过解析，仅从现有 chunks.json 重建索引
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
                        help="跳过解析，仅从现有 chunks.json 重建索引")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：.env 中未设置 OPENAI_API_KEY")
        sys.exit(1)

    print("=" * 60)
    print("特斯拉财报摄取流程")
    print("=" * 60)

    if args.index_only:
        if not CHUNKS_PATH.exists():
            print(f"错误：未找到 {CHUNKS_PATH}。请先运行完整流程（不带 --index-only）。")
            sys.exit(1)
        print(f"\n[跳过解析] 加载现有 {CHUNKS_PATH}...")
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        print(f"已加载 {len(chunks)} 个块")
    else:
        from src.json_parser import parse_all_jsons
        from src.chunker import merge_and_rechunk

        json_files = list(DATASET_DIR.rglob("*.json"))
        print(f"\n[1/3] 解析 {len(json_files)} 个 JSON 文件...")
        raw_chunks = parse_all_jsons(DATASET_DIR)
        print(f"原始块总数：{len(raw_chunks)}")

        print("\n[2/3] 语义分块...")
        chunks = merge_and_rechunk(raw_chunks)
        print(f"最终块总数：{len(chunks)}")

        years = sorted(set(c.get("year") for c in chunks))
        periods = sorted(set(c.get("period") for c in chunks))
        tables = sum(1 for c in chunks if c.get("chunk_type") == "table")
        texts  = sum(1 for c in chunks if c.get("chunk_type") == "text")
        quarterly = sum(1 for c in chunks if c.get("doc_type") == "quarterly_report")
        annual    = sum(1 for c in chunks if c.get("doc_type") == "annual_report")
        print(f"  年份：{years}")
        print(f"  期间数：{len(periods)}")
        print(f"  表格块：{tables} | 文本块：{texts}")
        print(f"  季度报告块：{quarterly} | 年度报告块：{annual}")

        CHUNKS_PATH.parent.mkdir(exist_ok=True)
        with open(CHUNKS_PATH, "w") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n[3/3] 构建向量 + BM25 索引...")
    from src.indexer import build_index
    build_index(chunks, api_key)

    print("\n✓ 摄取完成！")


if __name__ == "__main__":
    main()
