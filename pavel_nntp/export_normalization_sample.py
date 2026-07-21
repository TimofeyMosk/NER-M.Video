from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import duckdb

from process_query_clicks import (
    MANIFEST_PATH,
    QUERY_CLICKS_PATH,
    REPORTS_DIR,
    normalize_text,
    write_manifest,
)


SAMPLE_SIZE = 1000
SEED = 42
CSV_PATH = REPORTS_DIR / "query_normalization_1000.csv"
HTML_PATH = REPORTS_DIR / "query_normalization_1000.html"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    query_path = str(QUERY_CLICKS_PATH).replace("'", "''")

    connection = duckdb.connect()
    try:
        originals = [
            row[0]
            for row in connection.execute(
                f"""
                SELECT query_original
                FROM (
                    SELECT DISTINCT CAST("toValidUTF8(query_text)" AS VARCHAR) AS query_original
                    FROM read_parquet('{query_path}')
                    WHERE "toValidUTF8(query_text)" IS NOT NULL
                      AND TRIM(CAST("toValidUTF8(query_text)" AS VARCHAR)) <> ''
                )
                ORDER BY MD5(query_original || '|{SEED}')
                LIMIT {SAMPLE_SIZE}
                """
            ).fetchall()
        ]
    finally:
        connection.close()

    rows = [(original, normalize_text(original)) for original in originals]

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query_original", "query_normalized"])
        writer.writerows(rows)

    changed = sum(original != normalized for original, normalized in rows)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(original)}</td>"
        f"<td>{html.escape(normalized)}</td>"
        "</tr>"
        for index, (original, normalized) in enumerate(rows, 1)
    )
    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1000 примеров нормализации запросов</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #20252b; font-family: Inter, system-ui, sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #596572; line-height: 1.5; }}
    .metrics {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 20px 0; }}
    .metric {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 10px 14px; }}
    .metric strong {{ display: block; font-size: 21px; }}
    input {{ width: min(520px, 100%); box-sizing: border-box; padding: 10px 12px; border: 1px solid #bfc7d0; border-radius: 6px; font-size: 15px; margin-bottom: 14px; }}
    .table-wrap {{ max-height: 72vh; overflow: auto; background: #fff; border: 1px solid #d9dee4; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ padding: 9px 11px; border-bottom: 1px solid #e5e8ec; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ position: sticky; top: 0; background: #eef1f4; z-index: 1; font-size: 12px; }}
    th:first-child, td:first-child {{ width: 54px; text-align: right; color: #687480; }}
    tr:hover td {{ background: #f8fafb; }}
  </style>
</head>
<body><main>
  <h1>1000 примеров нормализации запросов</h1>
  <p>Показана базовая очистка до словарной канонизации. Выборка уникальных запросов воспроизводима с seed {SEED}.</p>
  <div class="metrics">
    <div class="metric"><strong>{len(rows)}</strong>запросов</div>
    <div class="metric"><strong>{changed}</strong>изменились</div>
    <div class="metric"><strong>{len(rows) - changed}</strong>уже были нормализованы</div>
  </div>
  <input id="search" type="search" placeholder="Поиск по таблице">
  <div class="table-wrap"><table>
    <thead><tr><th>№</th><th>query_original</th><th>query_normalized</th></tr></thead>
    <tbody id="rows">{table_rows}</tbody>
  </table></div>
  <script>
    const input = document.getElementById('search');
    const rows = [...document.querySelectorAll('#rows tr')];
    input.addEventListener('input', () => {{
      const value = input.value.toLowerCase();
      rows.forEach(row => row.hidden = !row.textContent.toLowerCase().includes(value));
    }});
  </script>
</main></body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")

    if MANIFEST_PATH.exists():
        metrics = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["metrics"]
        write_manifest(metrics)

    print(f"Создано: {CSV_PATH}")
    print(f"Создано: {HTML_PATH}")


if __name__ == "__main__":
    main()
