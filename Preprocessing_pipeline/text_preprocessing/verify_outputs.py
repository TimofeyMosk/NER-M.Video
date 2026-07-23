#!/usr/bin/env python3
"""Verify generated preprocessing artifacts against their metrics manifest."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> int:
    metrics = json.loads((OUTPUT / "preprocessing_metrics.json").read_text(encoding="utf-8"))
    counts = metrics["counts"]
    audit = OUTPUT / "preprocessed_queries_audit.parquet"
    final = OUTPUT / "preprocessed_queries.parquet"
    report = OUTPUT / "text_preprocessing_report.html"
    connection = duckdb.connect()

    audit_row = connection.execute(
        "SELECT COUNT(*), SUM(raw_count), "
        "COUNT(DISTINCT query_original), COUNT(DISTINCT nfkc), COUNT(DISTINCT yo_equal_e), "
        "COUNT(DISTINCT lower), COUNT(DISTINCT separators), COUNT(DISTINCT spaces), "
        "COUNT(DISTINCT CASE WHEN lemma <> '' THEN lemma END), "
        "COALESCE(SUM(raw_count) FILTER (WHERE lemma = ''), 0) "
        "FROM read_parquet(?)",
        [str(audit)],
    ).fetchone()
    final_row = connection.execute(
        "SELECT COUNT(*), SUM(source_row_count), COUNT(DISTINCT query_preprocessed), "
        "COUNT(*) FILTER (WHERE query_preprocessed = '' OR query_preprocessed IS NULL) "
        "FROM read_parquet(?)",
        [str(final)],
    ).fetchone()
    connection.close()

    expected_audit = (
        counts["raw_unique"], counts["raw_rows"], counts["raw_unique"], counts["nfkc_unique"],
        counts["yo_unique"], counts["lower_unique"], counts["separator_unique"],
        counts["space_unique"], counts["lemma_unique"], counts["empty_final"],
    )
    if tuple(int(value) for value in audit_row) != tuple(int(value) for value in expected_audit):
        raise AssertionError(f"audit mismatch: actual={audit_row}, expected={expected_audit}")
    if int(final_row[0]) != counts["lemma_unique"]:
        raise AssertionError(f"final row count {final_row[0]} != {counts['lemma_unique']}")
    if int(final_row[0]) != int(final_row[2]):
        raise AssertionError("final query_preprocessed is not unique")
    if int(final_row[3]) != 0:
        raise AssertionError("final output contains empty queries")
    if int(final_row[1]) != counts["raw_rows"] - counts["empty_final"]:
        raise AssertionError("final source_row_count does not reconcile to source")
    report_text = report.read_text(encoding="utf-8")
    for expected in ("Предобработка поисковых запросов", "Что работает", "Что не решено"):
        if expected not in report_text:
            raise AssertionError(f"report section missing: {expected}")

    print("PASSED")
    print(f"audit rows: {audit_row[0]:,}")
    print(f"final unique rows: {final_row[0]:,}")
    print(f"source rows represented: {final_row[1]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
