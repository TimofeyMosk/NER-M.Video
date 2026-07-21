#!/usr/bin/env python3
"""Validate the generated silver annotation dataset and its shards."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


OUTPUT = Path(__file__).resolve().parent / "output"
DATASET = OUTPUT / "silver_labels.parquet"
SHARDS = OUTPUT / "shards"
METRICS = OUTPUT / "annotation_metrics.json"
REVIEW = OUTPUT / "manual_review_sample.csv"
REPORT = OUTPUT / "ANNOTATION_REPORT.md"


def main() -> int:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    connection = duckdb.connect()
    summary = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT annotation_id), COUNT(DISTINCT query_preprocessed), "
        "COUNT(*) FILTER (WHERE query_preprocessed='' OR query_preprocessed IS NULL), "
        "COUNT(*) FILTER (WHERE category_id IS NULL OR category IS NULL), "
        "MIN(category_confidence), MIN(click_count), "
        "COUNT(*) FILTER (WHERE entity_labels_complete), "
        "COUNT(*) FILTER (WHERE NOT category_label_complete), "
        "COUNT(*) FILTER (WHERE split <> 'validation') "
        "FROM read_parquet(?)",
        [str(DATASET)],
    ).fetchone()
    rows = int(summary[0])
    if rows != int(metrics["rows"]):
        raise AssertionError(f"metrics rows={metrics['rows']}, parquet rows={rows}")
    if int(summary[1]) != rows or int(summary[2]) != rows:
        raise AssertionError("annotation_id and query_preprocessed must be unique")
    if int(summary[3]) or int(summary[4]):
        raise AssertionError("empty query or missing category detected")
    if float(summary[5]) < 0.80 or int(summary[6]) < 3:
        raise AssertionError("category quality gate violated")
    if int(summary[7]) != 0 or int(summary[8]) != 0:
        raise AssertionError("label completeness flags are inconsistent")
    if int(summary[9]) != 0:
        raise AssertionError(f"{summary[9]} rows are not marked as validation")
    bad_grades = connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?) WHERE "
        "(quality_grade='A' AND NOT (category_confidence >= 0.95 AND click_count >= 5)) OR "
        "(quality_grade='B' AND NOT (category_confidence >= 0.90 AND category_confidence < 0.95 AND click_count >= 5)) OR "
        "(quality_grade='C' AND (category_confidence >= 0.90 AND click_count >= 5)) OR "
        "quality_grade NOT IN ('A', 'B', 'C')",
        [str(DATASET)],
    ).fetchone()[0]
    if bad_grades:
        raise AssertionError(f"{bad_grades} rows violate quality grade rules")

    source_overlap = connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?) labels "
        "LEFT JOIN read_parquet(?) source USING (query_preprocessed) "
        "WHERE source.query_preprocessed IS NULL",
        [str(DATASET), str(Path(__file__).resolve().parents[1] / "text_preprocessing" / "output" / "preprocessed_queries.parquet")],
    ).fetchone()[0]
    if source_overlap:
        raise AssertionError(f"{source_overlap} labelled queries are absent from preprocessing corpus")

    shard_paths = sorted(SHARDS.glob("part-*.parquet"))
    if len(shard_paths) != int(metrics["shards"]):
        raise AssertionError("unexpected shard count")
    shard_total = 0
    for index, path in enumerate(shard_paths):
        shard_rows, bad_shard = connection.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE shard <> ?) FROM read_parquet(?)",
            [index, str(path)],
        ).fetchone()
        if bad_shard:
            raise AssertionError(f"wrong shard marker in {path.name}")
        shard_total += int(shard_rows)
    if shard_total != rows:
        raise AssertionError(f"shards sum to {shard_total}, expected {rows}")

    invalid_spans = 0
    invalid_bio = 0
    invalid_confidence = 0
    entity_count = 0
    for query, entities_json, tokens_json, tags_json, mask_json in connection.execute(
        "SELECT query_normalized, entities_json, tokens_json, bio_tags_weak_json, bio_supervision_mask_json "
        "FROM read_parquet(?)",
        [str(DATASET)],
    ).fetchall():
        entities = json.loads(entities_json)
        tokens = json.loads(tokens_json)
        tags = json.loads(tags_json)
        mask = json.loads(mask_json)
        entity_count += len(entities)
        if not (len(tokens) == len(tags) == len(mask)):
            invalid_bio += 1
        if any((tag == "O") != (not supervised) for tag, supervised in zip(tags, mask)):
            invalid_bio += 1
        occupied: set[int] = set()
        for entity in entities:
            start, end = int(entity["start"]), int(entity["end"])
            if not (0 <= start < end <= len(query)) or query[start:end] != entity["text"]:
                invalid_spans += 1
            token_positions = set(range(int(entity["token_start"]), int(entity["token_end"])))
            if token_positions & occupied:
                invalid_spans += 1
            occupied.update(token_positions)
            minimum = 0.90 if entity["type"] == "brand" else 0.80
            if float(entity["confidence"]) < minimum:
                invalid_confidence += 1
    connection.close()
    if invalid_spans or invalid_bio or invalid_confidence:
        raise AssertionError(
            f"invalid_spans={invalid_spans}, invalid_bio={invalid_bio}, "
            f"invalid_confidence={invalid_confidence}"
        )
    if entity_count != int(metrics["entity_total"]):
        raise AssertionError(f"entity count {entity_count} != metrics {metrics['entity_total']}")

    with REVIEW.open(encoding="utf-8", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    if len(review_rows) != 1000:
        raise AssertionError(f"manual review sample has {len(review_rows)} rows, expected 1000")
    required_review = {"annotation_id", "reviewer_status", "reviewer_comment", "entities_json"}
    if not required_review.issubset(review_rows[0]):
        raise AssertionError("manual review columns are incomplete")
    report = REPORT.read_text(encoding="utf-8")
    for heading in ("# Отчёт о silver-разметке", "## Контроль качества", "## Ограничения и риски", "## Уточнение по уровням качества"):
        if heading not in report:
            raise AssertionError(f"missing report section: {heading}")

    print("PASSED")
    print(f"rows={rows:,}, entities={entity_count:,}, shards={len(shard_paths)}, review={len(review_rows):,}")
    print(f"category_confidence_min={float(summary[5]):.3f}, click_count_min={int(summary[6])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
