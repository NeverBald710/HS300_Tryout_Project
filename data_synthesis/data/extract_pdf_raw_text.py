# 一键流程：先抽样巡检坏 PDF，删除坏文件，再全量并行转 TXT
# python data_synthesis/data/extract_pdf_raw_text.py --sample-pages 8 --workers 6 --use-cache

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pypdf import PdfReader

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "raw"
PDF_DIR = DATA_DIR / "pdf"
TEXT_DIR = DATA_DIR / "text"
META_FILE = DATA_DIR / "metadata.csv"
CACHE_FILE = DATA_DIR / "qc_cache.jsonl"
VALID_PUNCT = set("，。！？；：、“”‘’（）《》【】\\-—,.!?;:\"")


def ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def p(path: Path) -> str:
    return str(path).replace("\\", "/")


def gibberish_ratio(text: str) -> float:
    if not text:
        return 1.0
    valid = 0
    for ch in text:
        if ch.isspace() or (0x4E00 <= ord(ch) <= 0x9FFF) or (ch.isascii() and ch.isalnum()) or ch in VALID_PUNCT:
            valid += 1
    return round(max(0.0, min(1.0, 1.0 - valid / len(text))), 6)


def extract_page_text(reader: PdfReader, page_index: int) -> str:
    return (reader.pages[page_index].extract_text() or "").strip()


def ocr_page(pdf_path: Path, page_index: int, dpi: int, lang: str) -> str:
    if pytesseract is None or convert_from_path is None:
        return ""
    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_index + 1, last_page=page_index + 1, thread_count=1)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=lang).strip()


def build_sample_indices(total: int, sample_pages: int, sample_every: int) -> list[int]:
    if total <= 0:
        return []
    idx: set[int] = {0, total - 1, total // 2}
    if total > 2:
        idx.add(1)
    if total > 3:
        idx.add(2)
    if sample_pages > len(idx):
        step = (total - 1) / max(1, sample_pages - 1)
        for i in range(sample_pages):
            idx.add(int(round(i * step)))
    if sample_every > 0:
        idx.update(range(0, total, sample_every))
    return sorted(i for i in idx if 0 <= i < total)


def extract_indices(reader: PdfReader, pdf_path: Path, indices: list[int], ocr_fallback: bool, ocr_lang: str, ocr_dpi: int) -> tuple[list[tuple[int, str]], int]:
    pages: list[tuple[int, str]] = []
    ocr_pages = 0
    for i in indices:
        text = extract_page_text(reader, i)
        if ocr_fallback and len(text) < 20:
            t2 = ocr_page(pdf_path, i, ocr_dpi, ocr_lang)
            if len(t2) > len(text):
                text = t2
                if text:
                    ocr_pages += 1
        pages.append((i + 1, text))
    return pages, ocr_pages


def write_text(path: Path, pages: list[tuple[int, str]]) -> None:
    lines: list[str] = []
    for page_no, text in pages:
        lines.extend([f"===== PAGE {page_no} START =====", text, f"===== PAGE {page_no} END =====", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cache_key(pdf_path: Path) -> str:
    st = pdf_path.stat()
    return f"{p(pdf_path.relative_to(DATA_DIR))}|{st.st_size}|{st.st_mtime_ns}"


def build_row(pdf_path: Path) -> dict[str, Any]:
    rel_pdf = pdf_path.relative_to(DATA_DIR)
    txt_path = (TEXT_DIR / rel_pdf.relative_to("pdf")).with_suffix(".txt")
    st = pdf_path.stat()
    return {
        "pdf_path": p(rel_pdf),
        "text_path": p(txt_path.relative_to(DATA_DIR)),
        "file_name": pdf_path.name,
        "company": pdf_path.parent.name,
        "date": "",
        "file_size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "pages": 0,
        "sampled_pages": 0,
        "stage": "",
        "suspect": False,
        "extract_status": "",
        "error": "",
        "text_length": 0,
        "gibberish_ratio": 1.0,
        "ocr_pages": 0,
        "cache_key": cache_key(pdf_path),
    }


def is_bad_pdf(sample_text: str, sampled_pages: int, total_pages: int) -> bool:
    return total_pages <= 0 or sampled_pages <= 0 or len(sample_text) < 40 or gibberish_ratio(sample_text) >= 0.55


def inspect_pdf(pdf_path: Path, ocr_fallback: bool, ocr_lang: str, ocr_dpi: int, sample_pages: int, sample_every: int) -> dict[str, Any]:
    row = build_row(pdf_path)
    try:
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        idx = build_sample_indices(total, max(1, sample_pages), max(0, sample_every))
        sampled_pages_text, sample_ocr = extract_indices(reader, pdf_path, idx, ocr_fallback, ocr_lang, ocr_dpi)
        sample_text = "\n".join(text for _, text in sampled_pages_text)
        row.update({
            "pages": total,
            "sampled_pages": len(idx),
            "stage": "sample_check",
            "suspect": is_bad_pdf(sample_text, len(idx), total),
            "extract_status": "sample_checked",
            "text_length": len(sample_text),
            "gibberish_ratio": gibberish_ratio(sample_text),
            "ocr_pages": sample_ocr,
        })
    except Exception as e:
        row.update({"stage": "sample_check", "suspect": True, "extract_status": "error", "error": str(e)})
    return row


def process_pdf(pdf_path: Path, ocr_fallback: bool, ocr_lang: str, ocr_dpi: int) -> dict[str, Any]:
    row = build_row(pdf_path)
    txt_path = (TEXT_DIR / pdf_path.relative_to(DATA_DIR).relative_to("pdf")).with_suffix(".txt")
    try:
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        pages, ocr_pages = extract_indices(reader, pdf_path, list(range(total)), ocr_fallback, ocr_lang, ocr_dpi)
        write_text(txt_path, pages)
        full_text = "\n".join(text for _, text in pages)
        row.update({
            "pages": total,
            "sampled_pages": total,
            "stage": "full",
            "extract_status": "ok" if full_text else "empty",
            "text_length": len(full_text),
            "gibberish_ratio": gibberish_ratio(full_text),
            "ocr_pages": ocr_pages,
        })
    except Exception as e:
        row.update({"stage": "full", "extract_status": "error", "error": str(e)})
    return row


def list_pdf_files(limit: int | None = None) -> list[Path]:
    files = sorted(PDF_DIR.rglob("*.pdf"))
    return files[:limit] if limit is not None else files


def save_metadata(rows: list[dict[str, Any]]) -> None:
    fields = ["pdf_path", "text_path", "file_name", "company", "date", "file_size", "mtime_ns", "pages", "sampled_pages", "stage", "suspect", "extract_status", "error", "text_length", "gibberish_ratio", "ocr_pages"]
    with META_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("cache_key") and isinstance(obj.get("row"), dict):
                    out[obj["cache_key"]] = obj["row"]
            except json.JSONDecodeError:
                pass
    return out


def save_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            key = row.get("cache_key")
            if key:
                cached = dict(row)
                cached.pop("cache_key", None)
                f.write(json.dumps({"cache_key": key, "row": cached}, ensure_ascii=False) + "\n")


def delete_bad_pdf(pdf_path: Path) -> None:
    txt_path = (TEXT_DIR / pdf_path.relative_to(DATA_DIR).relative_to("pdf")).with_suffix(".txt")
    if txt_path.exists():
        txt_path.unlink()
    pdf_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="先抽样检测坏 PDF，删除坏文件，再全量并行提取 TXT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--ocr-lang", default="chi_sim+eng")
    parser.add_argument("--ocr-dpi", type=int, default=250)
    parser.add_argument("--sample-pages", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-file", default=str(CACHE_FILE))
    args = parser.parse_args()

    ensure_dirs()
    pdf_files = list_pdf_files(args.limit)
    if not pdf_files:
        print(f"未找到 PDF 文件，请放入: {PDF_DIR}")
        return

    total = len(pdf_files)
    print(f"开始抽样巡检，共 {total} 个 PDF")
    sample_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), total)) as ex:
        futures = [ex.submit(inspect_pdf, pdf, not args.no_ocr, args.ocr_lang, args.ocr_dpi, args.sample_pages, args.sample_every) for pdf in pdf_files]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            sample_rows.append(row)
            print(f"[抽样 {i}/{total}] {row['file_name']} | suspect={row['suspect']} 页数={row['pages']} 抽样页={row['sampled_pages']} 长度={row['text_length']} 乱码比={row['gibberish_ratio']}")
            if row.get("error"):
                print(f"  错误: {row['error']}")

    row_map = {row["pdf_path"]: row for row in sample_rows}
    good_pdfs: list[Path] = []
    deleted_rows: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        row = row_map[p(pdf_path.relative_to(DATA_DIR))]
        if row["suspect"]:
            delete_bad_pdf(pdf_path)
            deleted = dict(row)
            deleted.update({"stage": "deleted", "extract_status": "deleted_bad_file"})
            deleted_rows.append(deleted)
        else:
            good_pdfs.append(pdf_path)

    print(f"抽样完成，删除坏文件 {len(deleted_rows)} 个，剩余 {len(good_pdfs)} 个")

    cache = load_cache(Path(args.cache_file)) if args.use_cache else {}
    full_rows: list[dict[str, Any]] = []
    pending: list[Path] = []
    for pdf_path in good_pdfs:
        key = cache_key(pdf_path)
        if args.use_cache and key in cache:
            row = dict(cache[key])
            row["cache_key"] = key
            full_rows.append(row)
        else:
            pending.append(pdf_path)

    if args.use_cache:
        print(f"全量阶段缓存命中 {len(full_rows)} / {len(good_pdfs)}")

    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(pending))) as ex:
            futures = [ex.submit(process_pdf, pdf, not args.no_ocr, args.ocr_lang, args.ocr_dpi) for pdf in pending]
            total_full = len(good_pdfs)
            done = len(full_rows)
            for future in as_completed(futures):
                done += 1
                row = future.result()
                full_rows.append(row)
                print(f"[全量 {done}/{total_full}] {row['file_name']} | 状态={row['extract_status']} 页数={row['pages']} 长度={row['text_length']} 乱码比={row['gibberish_ratio']}")
                if row.get("error"):
                    print(f"  错误: {row['error']}")

    rows = sorted(deleted_rows + full_rows, key=lambda x: x.get("pdf_path", ""))
    save_metadata(rows)
    if args.use_cache:
        save_cache(Path(args.cache_file), full_rows)
    print(f"\n完成，metadata 已写入: {META_FILE}")


if __name__ == "__main__":
    main()
