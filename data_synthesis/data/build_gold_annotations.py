from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "raw"
CHUNK_FILE = DATA_DIR / "chunks" / "financial_sections.jsonl"
OUTPUT_DIR = DATA_DIR / "gold"
OUTPUT_FILE = OUTPUT_DIR / "gold_annotation_samples.jsonl"

DEFAULT_SCHEMA = {
    "company_name": None,
    "report_year": None,
    "report_type": None,
    "revenue": None,
    "revenue_yoy": None,
    "net_profit": None,
    "net_profit_yoy": None,
    "operating_cashflow": None,
    "risk_alerts": [],
    "evidence_spans": [],
}

PREFERRED_SECTION_ORDER = [
    "会计数据和财务指标",
    "管理层讨论与分析",
    "风险管理",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_chunk(row: dict[str, Any]) -> tuple[int, int]:
    text = row.get("text", "")
    section_name = row.get("section_name", "")
    base = 0
    if section_name == "会计数据和财务指标":
        for key in ["营业收入", "净利润", "经营活动产生的现金流量净额", "同比"]:
            if key in text:
                base += 3
    elif section_name == "管理层讨论与分析":
        for key in ["营业收入", "净利润", "资产总额", "风险", "不良贷款率"]:
            if key in text:
                base += 2
    elif section_name == "风险管理":
        for key in ["风险", "不良贷款率", "拨备覆盖率", "资产质量"]:
            if key in text:
                base += 2
    return base, len(text)


def deduplicate_chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("file_name", "")),
            str(row.get("section_name", "")),
            str(row.get("text", ""))[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def grouped_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = deduplicate_chunks(rows)

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_section[str(row.get("section_name", "其他"))].append(row)

    for section_rows in by_section.values():
        section_rows.sort(key=score_chunk, reverse=True)

    ordered_sections = [name for name in PREFERRED_SECTION_ORDER if name in by_section]
    ordered_sections += sorted(name for name in by_section if name not in ordered_sections)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # 第一轮：每个 section 先拿一部分高质量样本
    if ordered_sections:
        base_take = max(1, sample_size // len(ordered_sections))
        for section_name in ordered_sections:
            pool = by_section[section_name]
            take = min(base_take, len(pool))
            for row in pool[:take]:
                chunk_id = str(row.get("chunk_id", ""))
                if chunk_id and chunk_id not in selected_ids:
                    selected.append(row)
                    selected_ids.add(chunk_id)

    # 第二轮：补足剩余名额，优先高分但尽量打散文件
    remainder_pool = sorted(rows, key=score_chunk, reverse=True)
    file_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        file_counts[str(row.get("file_name", ""))] += 1

    for row in remainder_pool:
        if len(selected) >= sample_size:
            break
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id or chunk_id in selected_ids:
            continue
        file_name = str(row.get("file_name", ""))
        if file_counts[file_name] >= 3:
            continue
        selected.append(row)
        selected_ids.add(chunk_id)
        file_counts[file_name] += 1

    # 第三轮：如果还不够，就随机补齐
    if len(selected) < sample_size:
        remaining = [row for row in rows if str(row.get("chunk_id", "")) not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining:
            if len(selected) >= sample_size:
                break
            selected.append(row)
            selected_ids.add(str(row.get("chunk_id", "")))

    return selected[:sample_size]


def build_annotation_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("chunk_id"),
        "source": {
            "stock_code": row.get("stock_code"),
            "company_name": row.get("company_name"),
            "report_year": row.get("report_year"),
            "report_type": row.get("report_type"),
            "file_name": row.get("file_name"),
            "text_path": row.get("text_path"),
            "section_name": row.get("section_name"),
            "section_title": row.get("section_title"),
            "chunk_index": row.get("chunk_index"),
            "char_count": row.get("char_count"),
        },
        "text": row.get("text", ""),
        "annotation": DEFAULT_SCHEMA.copy(),
        "notes": "手工标注：仅填写文本中明确出现的信息；没有明确提到的字段保持 null；risk_alerts 为字符串数组；evidence_spans 填原文短句数组。",
    }


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="从财报 chunk 中抽样生成黄金集标注模板")
    parser.add_argument("--input", type=str, default=str(CHUNK_FILE), help="输入 chunk jsonl 文件")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="输出标注模板 jsonl 文件")
    parser.add_argument("--sample-size", type=int, default=120, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT_DIR / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    rows = load_chunks(input_path)
    selected = grouped_sample(rows, sample_size=args.sample_size, seed=args.seed)
    annotated = [build_annotation_record(row) for row in selected]
    save_jsonl(output_path, annotated)

    print(f"输入 chunks: {len(rows)}")
    print(f"输出样本: {len(annotated)}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
