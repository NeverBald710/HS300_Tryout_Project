from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import Iterable

import requests

try:
    import akshare as ak
except ImportError:
    ak = None


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "raw"
PDF_DIR = DATA_DIR / "pdf"
META_DIR = DATA_DIR / "metadata"

CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_PREFIX = "https://static.cninfo.com.cn/"
DEFAULT_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://www.cninfo.com.cn",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}

TARGET_KEYWORDS = ("年度报告", "半年度报告")
EXCLUDE_KEYWORDS = (
    "摘要",
    "英文",
    "取消",
    "更正",
    "修订",
    "更新后",
    "补充",
    "提示性",
)


def ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def recent_two_year_range() -> tuple[str, str]:
    today = dt.date.today()
    start = dt.date(today.year - 2, 1, 1)
    return start.isoformat(), today.isoformat()


def normalize_stock_code(code: str) -> str:
    return str(code).strip().zfill(6)


def infer_exchange(code: str) -> tuple[str, str]:
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh", "sse"
    return "sz", "szse"


def get_hs300_constituents() -> list[dict[str, str]]:
    if ak is None:
        raise ImportError("缺少 akshare，请先执行: pip install akshare requests")

    df = ak.index_stock_cons_csindex(symbol="000300")
    column_candidates = {
        "code": ["成分券代码", "证券代码", "股票代码", "code"],
        "name": ["成分券名称", "证券简称", "股票简称", "name"],
    }

    def pick_column(candidates: Iterable[str]) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"未找到列: {list(candidates)}，实际列为: {list(df.columns)}")

    code_col = pick_column(column_candidates["code"])
    name_col = pick_column(column_candidates["name"])

    return [
        {
            "stock_code": normalize_stock_code(row[code_col]),
            "company_name": str(row[name_col]).strip(),
        }
        for _, row in df[[code_col, name_col]].drop_duplicates().iterrows()
    ]


def should_keep_title(title: str) -> bool:
    if not any(keyword in title for keyword in TARGET_KEYWORDS):
        return False
    if any(keyword in title for keyword in EXCLUDE_KEYWORDS):
        return False
    return True


def build_pdf_url(adjunct_url: str) -> str:
    adjunct_url = adjunct_url.strip()
    if adjunct_url.startswith(("http://", "https://")):
        return adjunct_url
    return CNINFO_STATIC_PREFIX + adjunct_url.lstrip("/")


def build_payload_candidates(
    stock_code: str,
    company_name: str,
    start_date: str,
    end_date: str,
    page_num: int,
    page_size: int,
) -> list[tuple[str, dict[str, str]]]:
    plate, column = infer_exchange(stock_code)
    base = {
        "pageNum": str(page_num),
        "pageSize": str(page_size),
        "column": column,
        "tabName": "fulltext",
        "trade": "",
        "seDate": f"{start_date}~{end_date}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

    return [
        (
            "stock=code,name + category",
            {
                **base,
                "plate": plate,
                "stock": f"{stock_code},{company_name}",
                "searchkey": "",
                "secid": "",
                "category": "category_ndbg_szsh;category_bndbg_szsh",
            },
        ),
        (
            "stock=code + category",
            {
                **base,
                "plate": plate,
                "stock": stock_code,
                "searchkey": "",
                "secid": "",
                "category": "category_ndbg_szsh;category_bndbg_szsh",
            },
        ),
        (
            "stock=code + empty plate + category",
            {
                **base,
                "plate": "",
                "stock": stock_code,
                "searchkey": "",
                "secid": "",
                "category": "category_ndbg_szsh;category_bndbg_szsh",
            },
        ),
        (
            "searchkey=code + category",
            {
                **base,
                "plate": plate,
                "stock": "",
                "searchkey": stock_code,
                "secid": "",
                "category": "category_ndbg_szsh;category_bndbg_szsh",
            },
        ),
        (
            "searchkey=name + category",
            {
                **base,
                "plate": plate,
                "stock": "",
                "searchkey": company_name,
                "secid": "",
                "category": "category_ndbg_szsh;category_bndbg_szsh",
            },
        ),
        (
            "stock=code + no category",
            {
                **base,
                "plate": plate,
                "stock": stock_code,
                "searchkey": "",
                "secid": "",
                "category": "",
            },
        ),
    ]


def fetch_announcements_page(
    session: requests.Session,
    payload_candidates: list[tuple[str, dict[str, str]]],
) -> tuple[list[dict], dict[str, str] | None, str | None]:
    for label, payload in payload_candidates:
        response = session.post(CNINFO_QUERY_URL, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        announcements = data.get("announcements") or []
        if announcements:
            return announcements, payload, label
    return [], None, None


def query_reports_for_stock(
    session: requests.Session,
    stock_code: str,
    company_name: str,
    start_date: str,
    end_date: str,
    page_size: int = 30,
    sleep_seconds: float = 0.35,
) -> tuple[list[dict], str | None]:
    page_num = 1
    all_rows: list[dict] = []
    matched_payload: dict[str, str] | None = None
    matched_label: str | None = None

    while True:
        payload_candidates = build_payload_candidates(
            stock_code=stock_code,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            page_num=page_num,
            page_size=page_size,
        )

        if matched_payload is not None:
            payload_candidates = [(
                matched_label or "reused payload",
                {
                    **matched_payload,
                    "pageNum": str(page_num),
                    "pageSize": str(page_size),
                },
            )]

        announcements, used_payload, used_label = fetch_announcements_page(session, payload_candidates)
        if not announcements:
            break

        if matched_payload is None:
            matched_payload = used_payload
            matched_label = used_label

        for item in announcements:
            title = str(item.get("announcementTitle") or "").strip()
            if not should_keep_title(title):
                continue

            adjunct_url = str(item.get("adjunctUrl") or "").strip()
            if not adjunct_url:
                continue

            all_rows.append(
                {
                    "stock_code": stock_code,
                    "company_name": company_name,
                    "announcement_id": item.get("announcementId", ""),
                    "title": title,
                    "announcement_time": item.get("announcementTime", ""),
                    "report_type": "annual" if "年度报告" in title else "semiannual",
                    "pdf_url": build_pdf_url(adjunct_url),
                    "adjunct_url": adjunct_url,
                    "page_column": infer_exchange(stock_code)[1],
                }
            )

        if len(announcements) < page_size:
            break

        page_num += 1
        time.sleep(sleep_seconds)

    dedup: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        key = (str(row["announcement_time"]), str(row["title"]))
        dedup[key] = row
    return list(dedup.values()), matched_label


def sanitize_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "：")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "'")
        .replace("<", "《")
        .replace(">", "》")
        .replace("|", "_")
        .strip()
    )


def download_pdf(session: requests.Session, row: dict, overwrite: bool = False) -> Path:
    stock_dir = PDF_DIR / row["stock_code"]
    stock_dir.mkdir(parents=True, exist_ok=True)

    report_date = str(row["announcement_time"])[:10].replace("-", "") or "unknown"
    filename = f"{report_date}_{row['stock_code']}_{sanitize_filename(row['title'])}.pdf"
    pdf_path = stock_dir / filename

    if pdf_path.exists() and not overwrite:
        return pdf_path

    response = session.get(row["pdf_url"], timeout=60, stream=True)
    response.raise_for_status()
    with pdf_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
    return pdf_path


def save_metadata(rows: list[dict], start_date: str, end_date: str) -> Path:
    output_path = META_DIR / f"hs300_reports_{start_date}_to_{end_date}.csv"
    fieldnames = [
        "stock_code",
        "company_name",
        "announcement_id",
        "title",
        "announcement_time",
        "report_type",
        "pdf_url",
        "adjunct_url",
        "page_column",
        "pdf_local_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载沪深300最近2年的年报和半年报 PDF")
    parser.add_argument("--start-date", default=None, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 家公司，便于调试")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已下载 PDF")
    args = parser.parse_args()

    ensure_dirs()
    default_start, default_end = recent_two_year_range()
    start_date = args.start_date or default_start
    end_date = args.end_date or default_end

    constituents = get_hs300_constituents()
    if args.limit is not None:
        constituents = constituents[: args.limit]

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    collected_rows: list[dict] = []
    total_companies = len(constituents)
    for idx, item in enumerate(constituents, start=1):
        stock_code = item["stock_code"]
        company_name = item["company_name"]
        print(f"[{idx}/{total_companies}] 查询 {stock_code} {company_name}")
        try:
            rows, matched_label = query_reports_for_stock(
                session=session,
                stock_code=stock_code,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            print(f"  查询失败: {exc}")
            continue

        if matched_label:
            print(f"  命中的 payload 方案: {matched_label}")
        else:
            print("  命中的 payload 方案: 无")

        print(f"  命中 {len(rows)} 份报告")
        for row in rows:
            try:
                pdf_path = download_pdf(session, row, overwrite=args.overwrite)
                row["pdf_local_path"] = str(pdf_path)
                collected_rows.append(row)
                print(f"    已下载: {pdf_path.name}")
            except Exception as exc:
                print(f"    下载失败 {row['title']}: {exc}")
        time.sleep(0.3)

    meta_path = save_metadata(collected_rows, start_date, end_date)
    print(f"\n完成，共保存 {len(collected_rows)} 份 PDF")
    print(f"元数据文件: {meta_path}")


if __name__ == "__main__":
    main()
