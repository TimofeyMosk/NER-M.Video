from __future__ import annotations

import csv
import html
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cu_ws" / "query_clicks.parquet"
CSV_PATH = Path(__file__).resolve().parent / "query_normalization_100.csv"
HTML_PATH = Path(__file__).resolve().parent / "query_normalization_100.html"
QUERY_COLUMN = "toValidUTF8(query_text)"
SAMPLE_SIZE = 100
SEED = 42

DASHES_RE = re.compile(r"[-\u00ad\u058a\u05be\u1400\u1806\u2010-\u2015\u2212\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\u30a0\ufe31-\ufe32\ufe58\ufe63\uff0d]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().replace("ё", "е")
    value = DASHES_RE.sub(" ", value)

    cleaned: list[str] = []
    for char in value:
        if char.isspace():
            cleaned.append(" ")
        elif unicodedata.category(char)[0] in {"L", "N"}:
            cleaned.append(char)
        elif char in "+./":
            cleaned.append(char)
        else:
            cleaned.append(" ")

    return WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()


def sample_queries(parquet_file: pq.ParquetFile) -> list[str]:
    rng = random.Random(SEED)
    row_groups = list(range(parquet_file.num_row_groups))
    rng.shuffle(row_groups)

    selected: list[str] = []
    seen: set[str] = set()

    for row_group in row_groups:
        table = parquet_file.read_row_group(row_group, columns=[QUERY_COLUMN])
        values = [value for value in table.column(0).to_pylist() if value and value.strip()]
        rng.shuffle(values)

        added_from_group = 0
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            selected.append(value)
            added_from_group += 1
            if len(selected) == SAMPLE_SIZE:
                return selected
            if added_from_group == 5:
                break

    raise RuntimeError(f"Удалось собрать только {len(selected)} уникальных запросов")


def write_csv(rows: list[tuple[int, str, str]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["№", "Оригинал", "После обработки"])
        writer.writerows(rows)


def write_html(rows: list[tuple[int, str, str]]) -> None:
    changed = sum(original != normalized for _, original, normalized in rows)
    with_dash = sum(bool(DASHES_RE.search(original)) for _, original, _ in rows)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{number}</td>"
        f"<td>{html.escape(original)}</td>"
        f"<td>{html.escape(normalized)}</td>"
        "</tr>"
        for number, original, normalized in rows
    )
    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>100 примеров нормализации запросов</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f6f8; color: #20252b; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    p {{ margin: 0 0 22px; color: #56616d; line-height: 1.5; }}
    .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 22px; }}
    .stat {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 12px 16px; }}
    .stat strong {{ display: block; font-size: 22px; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #d9dee4; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ padding: 11px 13px; border-bottom: 1px solid #e5e8ec; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ position: sticky; top: 0; background: #eef1f4; font-size: 13px; }}
    th:first-child, td:first-child {{ width: 48px; text-align: right; color: #697580; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr:hover td {{ background: #f8fafb; }}
  </style>
</head>
<body>
<main>
  <h1>100 примеров нормализации запросов</h1>
  <p>Полная очистка: Unicode приводится к единому виду, текст переводится в нижний регистр, ё заменяется на е, все виды тире и дефиса заменяются пробелом, служебные символы удаляются, цифры и значимые символы +, . и / сохраняются, повторные пробелы схлопываются.</p>
  <div class="stats">
    <div class="stat"><strong>{len(rows)}</strong>уникальных запросов</div>
    <div class="stat"><strong>{changed}</strong>изменились</div>
    <div class="stat"><strong>{with_dash}</strong>содержали тире или дефис</div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>№</th><th>Оригинал</th><th>После обработки</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")


def main() -> None:
    parquet_file = pq.ParquetFile(DATA_PATH)
    queries = sample_queries(parquet_file)
    rows = [(index, query, normalize_query(query)) for index, query in enumerate(queries, 1)]
    write_csv(rows)
    write_html(rows)
    print(f"Создано: {HTML_PATH}")
    print(f"Создано: {CSV_PATH}")


if __name__ == "__main__":
    main()
