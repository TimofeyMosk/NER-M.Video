#!/usr/bin/env python3
"""Validate measurement candidates and positive-only BIO pre-annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


OUTPUT = Path(__file__).resolve().parent / "output"
AUDIT = OUTPUT / "preprocessed_queries_audit.parquet"
FINAL = OUTPUT / "preprocessed_queries.parquet"
METRICS = OUTPUT / "preprocessing_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    audit = output / AUDIT.name
    final = output / FINAL.name
    metrics_path = output / METRICS.name
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    parser_metrics = metrics["measurement_parser"]
    connection = duckdb.connect()
    columns = {
        row[0] for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(audit)]).fetchall()
    }
    required = {
        "query_model", "measurement_candidates_json", "measurement_bio_entities_json",
        "measurement_tokens_json", "measurement_bio_tags_json", "measurement_bio_mask_json",
        "measurement_rejected_count",
    }
    if not required.issubset(columns):
        raise AssertionError(f"missing measurement columns: {sorted(required - columns)}")

    summary = connection.execute(
        "SELECT COUNT(*), "
        "SUM(json_array_length(measurement_candidates_json)), "
        "SUM(json_array_length(measurement_bio_entities_json)), "
        "SUM(measurement_rejected_count), "
        "COUNT(*) FILTER (WHERE json_array_length(measurement_candidates_json) > 0), "
        "COUNT(*) FILTER (WHERE json_array_length(measurement_bio_entities_json) > 0) "
        "FROM read_parquet(?)",
        [str(audit)],
    ).fetchone()
    expected = (
        int(metrics["counts"]["raw_unique"]),
        int(parser_metrics["candidate_mentions"]),
        int(parser_metrics["bio_entities"]),
        int(parser_metrics["rejected_mentions"]),
        int(parser_metrics["queries_with_candidates"]),
        int(parser_metrics["queries_with_bio"]),
    )
    actual = tuple(int(value or 0) for value in summary)
    if actual != expected:
        raise AssertionError(f"measurement metric mismatch: actual={actual}, expected={expected}")

    invalid_spans = invalid_bio = invalid_policy = 0
    cursor = connection.execute(
        "SELECT query_model, measurement_bio_entities_json, measurement_tokens_json, "
        "measurement_bio_tags_json, measurement_bio_mask_json FROM read_parquet(?) "
        "WHERE json_array_length(measurement_bio_entities_json) > 0",
        [str(audit)],
    )
    while rows := cursor.fetchmany(5000):
        for query, entities_json, tokens_json, tags_json, mask_json in rows:
            entities = json.loads(entities_json)
            tokens = json.loads(tokens_json)
            tags = json.loads(tags_json)
            mask = json.loads(mask_json)
            if not (len(tokens) == len(tags) == len(mask)):
                invalid_bio += 1
                continue
            if any((tag == "O") != (not supervised) for tag, supervised in zip(tags, mask)):
                invalid_bio += 1
            for entity in entities:
                start, end = int(entity["start"]), int(entity["end"])
                if not (0 <= start < end <= len(query)) or query[start:end] != entity["surface"]:
                    invalid_spans += 1
                if not entity["bio_eligible"] or not entity["entity_type"] or not entity["has_number"]:
                    invalid_policy += 1
                if not any(
                    token["start"] < end and token["end"] > start and tag.endswith("-" + entity["entity_type"])
                    for token, tag in zip(tokens, tags)
                ):
                    invalid_bio += 1

    final_rows = int(connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(final)]).fetchone()[0])
    connection.close()
    if final_rows != int(metrics["counts"]["lemma_unique"]):
        raise AssertionError("final preprocessing row count mismatch")
    if invalid_spans or invalid_bio or invalid_policy:
        raise AssertionError(
            f"invalid_spans={invalid_spans}, invalid_bio={invalid_bio}, invalid_policy={invalid_policy}"
        )
    if parser_metrics["regular_expressions"]:
        raise AssertionError("measurement parser unexpectedly uses regular expressions")

    print("PASSED")
    print(
        f"queries={actual[0]:,}, candidates={actual[1]:,}, bio_entities={actual[2]:,}, "
        f"bio_queries={actual[5]:,}, rejected={actual[3]:,}, final_rows={final_rows:,}"
    )
    print(f"bio_types={parser_metrics['bio_entity_type_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
