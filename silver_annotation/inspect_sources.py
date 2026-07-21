#!/usr/bin/env python3
"""Inspect local Parquet sources used by the silver annotation pipeline."""

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
QUERY_DATASET = ROOT / "pavel_nntp" / "data" / "processed" / "query_dataset.parquet"
QUERY_SKU = ROOT / "pavel_nntp" / "data" / "processed" / "query_sku_aggregated.parquet"
ATTRIBUTES = ROOT / "pavel_nntp" / "data" / "processed" / "catalog_attributes.parquet"


def main() -> None:
    connection = duckdb.connect()
    for name, path in (("query_dataset", QUERY_DATASET), ("query_sku", QUERY_SKU), ("attributes", ATTRIBUTES)):
        print(f"\n=== {name} ===")
        for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall():
            print(row)
    print("\n=== quality ===")
    print(connection.execute(
        "SELECT quality, COUNT(*) FROM read_parquet(?) GROUP BY quality ORDER BY quality",
        [str(QUERY_DATASET)],
    ).fetchall())
    print("\n=== candidate thresholds ===")
    print(connection.execute(
        "SELECT "
        "COUNT(*) FILTER (WHERE category_confidence >= 0.80 AND click_count >= 3), "
        "COUNT(*) FILTER (WHERE category_confidence >= 0.90 AND click_count >= 5), "
        "COUNT(*) FILTER (WHERE category_confidence >= 0.95 AND click_count >= 5), "
        "COUNT(*) FILTER (WHERE category_confidence >= 0.95 AND click_count >= 10) "
        "FROM read_parquet(?)",
        [str(QUERY_DATASET)],
    ).fetchone())
    connection.close()


if __name__ == "__main__":
    main()
