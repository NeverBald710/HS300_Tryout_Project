from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "raw"
TEXT_DIR = DATA_DIR / "text"
META_FILE = DATA_DIR / "metadata.csv"
CHUNK_DIR = DATA_DIR / "chunks"
OUTPUT_FILE = CHUNK_DIR / "financial_sections.jsonl"

PAGE_MARKER_RE = re.compile(r"^===== PAGE \d+ (START|END) =====$")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零]+章\s+.+")
SECTION_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+.+")
REPORT_YEAR_RE = re.compile(r"(20\d{2})年")
REPORT_TYPE_RE = re.compile(r"(年度报告|半年度报告)")

TARGET_KEYWORDS = {
    "会计数据和财务指标": ["会计数据和财务指标", "主要会计数据", "主要财务指标", "财务摘要"],
    "管理层讨论与分析": ["管理层讨论与分析", "经营情况讨论与分析", "总体经营情况", "财务报表分析"],
    "风险管理": ["风险管理", "风险提示", "重大风险", "风险因素"],
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^第\s*\d+\s*页.*$"),
    re.compile(r"^目\s*录$"),
]


def ensure_dirs() -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> dict[str, dict[str, str]]:
    if not META_FILE.exists():
        raise FileNotFoundError(f"metadata 文件不存在: {META_FILE}")

    out: dict[str, dict[str, str]] = {}
    with META_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("extract_status") != "ok":
                continue
            text_path = (row.get("text_path") or "").replace("\\", "/")
            if text_path:
                out[text_path] = row
    return out


def clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if PAGE_MARKER_RE.match(line):
            continue
        if any(pattern.match(line) for pattern in NOISE_LINE_PATTERNS):
            continue
        lines.append(line)
    return lines


def is_heading(line: str) -> tuple[bool, int]:
    if not line or len(line) > 80:
        return False, 99

    if CHAPTER_RE.match(line):
        return True, 1

    section_match = SECTION_RE.match(line)
    if section_match:
        depth = section_match.group(1).count(".") + 2
        return True, depth

    if any(keyword in line for keys in TARGET_KEYWORDS.values() for keyword in keys):
        return True, 2

    return False, 99


def normalize_section_name(title: str) -> str | None:
    for canonical, keywords in TARGET_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            return canonical
    return None


def build_heading_index(lines: list[str]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        ok, level = is_heading(line)
        if not ok:
            continue
        canonical = normalize_section_name(line)
        headings.append(
            {
                "line_idx": idx,
                "title": line,
                "level": level,
                "canonical": canonical,
            }
        )
    return headings


def slice_sections(lines: list[str], headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for i, heading in enumerate(headings):
        if heading["canonical"] is None:
            continue

        start = heading["line_idx"]
        end = len(lines)
        current_level = heading["level"]

        for nxt in headings[i + 1 :]:
            if nxt["line_idx"] <= start:
                continue
            if nxt["level"] <= current_level:
                end = nxt["line_idx"]
                break

        body_lines = lines[start:end]
        body_text = "\n".join(body_lines).strip()
        if len(body_text) < 200:
            continue

        sections.append(
            {
                "section_title": heading["title"],
                "section_name": heading["canonical"],
                "text": body_text,
            }
        )
    return sections


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        if is_heading(stripped)[0] and current:
            paragraphs.append(" ".join(current))
            current = [stripped]
            continue

        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))
    return [p for p in paragraphs if p.strip()]


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            if overlap_chars > 0:
                overlap = current[-overlap_chars:]
                current = f"{overlap}\n\n{para}"
            else:
                current = para
        else:
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars - overlap_chars :] if overlap_chars < max_chars else para[max_chars:]
            current = para

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 120]


def infer_report_info(file_name: str) -> tuple[str | None, str | None]:
    year_match = REPORT_YEAR_RE.search(file_name)
    type_match = REPORT_TYPE_RE.search(file_name)
    report_year = year_match.group(1) if year_match else None
    report_type = type_match.group(1) if type_match else None
    return report_year, report_type


def build_records_for_file(
    txt_path: Path,
    meta: dict[str, str] | None,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = clean_lines(text)
    headings = build_heading_index(lines)
    sections = slice_sections(lines, headings)

    relative_text_path = txt_path.relative_to(DATA_DIR).as_posix()
    report_year, report_type = infer_report_info(txt_path.name)
    stock_code = txt_path.parent.name

    records: list[dict[str, Any]] = []
    for section in sections:
        chunks = chunk_text(section["text"], max_chars=max_chars, overlap_chars=overlap_chars)
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "stock_code": stock_code,
                    "company_name": meta.get("company", stock_code) if meta else stock_code,
                    "report_year": report_year,
                    "report_type": report_type,
                    "file_name": txt_path.name,
                    "text_path": relative_text_path,
                    "section_name": section["section_name"],
                    "section_title": section["section_title"],
                    "chunk_id": f"{txt_path.stem}::{section['section_name']}::{idx}",
                    "chunk_index": idx,
                    "char_count": len(chunk),
                    "text": chunk,
                }
            )
    return records


def list_text_files(limit: int | None = None) -> list[Path]:
    files = sorted(TEXT_DIR.rglob("*.txt"))
    return files[:limit] if limit is not None else files


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="切分财报 TXT 中的关键章节并输出训练 chunk")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 txt 文件，便于调试")
    parser.add_argument("--max-chars", type=int, default=1600, help="单个 chunk 最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=200, help="相邻 chunk 重叠字符数")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="输出 jsonl 路径")
    args = parser.parse_args()

    ensure_dirs()
    metadata = load_metadata()
    text_files = list_text_files(limit=args.limit)

    rows: list[dict[str, Any]] = []
    for idx, txt_path in enumerate(text_files, start=1):
        relative_text_path = txt_path.relative_to(DATA_DIR).as_posix()
        meta = metadata.get(relative_text_path)
        file_rows = build_records_for_file(
            txt_path=txt_path,
            meta=meta,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        rows.extend(file_rows)
        print(f"[{idx}/{len(text_files)}] {txt_path.name} -> {len(file_rows)} chunks")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    save_jsonl(output_path, rows)
    print(f"\n完成，共输出 {len(rows)} 条 chunk")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
