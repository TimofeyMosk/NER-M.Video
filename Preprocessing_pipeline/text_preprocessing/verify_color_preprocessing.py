#!/usr/bin/env python3
"""Validate color normalization, spans and positive-only BIO output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    audit = output / "preprocessed_queries_audit.parquet"
    metrics = json.loads((output / "preprocessing_metrics.json").read_text(encoding="utf-8"))
    expected_metrics = metrics["color_normalizer"]
    connection = duckdb.connect()
    columns = {
        row[0] for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(audit)]).fetchall()
    }
    required = {
        "query_model", "query_color_canonical", "color_candidates_json",
        "color_bio_entities_json", "color_tokens_json", "color_bio_tags_json",
        "color_bio_mask_json", "color_rejected_count",
    }
    if not required.issubset(columns):
        raise AssertionError(f"missing color columns: {sorted(required - columns)}")
    summary = connection.execute(
        "SELECT SUM(json_array_length(color_candidates_json)), SUM(color_rejected_count), "
        "COUNT(*) FILTER (WHERE json_array_length(color_candidates_json) > 0), "
        "COUNT(*) FILTER (WHERE query_color_canonical <> query_model) FROM read_parquet(?)",
        [str(audit)],
    ).fetchone()
    actual = tuple(int(value or 0) for value in summary)
    expected = (
        int(expected_metrics["candidate_mentions"]), int(expected_metrics["rejected_mentions"]),
        int(expected_metrics["queries_with_candidates"]), int(expected_metrics["queries_canonicalized"]),
    )
    if actual != expected:
        raise AssertionError(f"color metric mismatch: actual={actual}, expected={expected}")

    invalid_spans = invalid_bio = invalid_palette = 0
    palette = {
        "черный", "белый", "разноцветный", "прозрачный", "серый", "серебристый",
        "хром", "бежевый", "синий", "зеленый", "красный", "коричневый",
        "нержавеющая сталь", "голубой", "розовый", "оранжевый", "фиолетовый",
        "графитовый", "золотистый", "желтый", "янтарный", "бронзовый",
        "бирюзовый", "кремовый", "мятный", "песочный", "шоколадный",
        "сиреневый", "бордовый", "медный",
    }
    cursor = connection.execute(
        "SELECT query_model, color_bio_entities_json, color_tokens_json, color_bio_tags_json, "
        "color_bio_mask_json FROM read_parquet(?) WHERE json_array_length(color_bio_entities_json) > 0",
        [str(audit)],
    )
    while rows := cursor.fetchmany(5000):
        for query, entities_raw, tokens_raw, tags_raw, mask_raw in rows:
            entities, tokens = json.loads(entities_raw), json.loads(tokens_raw)
            tags, mask = json.loads(tags_raw), json.loads(mask_raw)
            if not (len(tokens) == len(tags) == len(mask)):
                invalid_bio += 1
                continue
            if any((tag == "O") != (not supervised) for tag, supervised in zip(tags, mask)):
                invalid_bio += 1
            for entity in entities:
                start, end = int(entity["start"]), int(entity["end"])
                if not (0 <= start < end <= len(query)) or query[start:end] != entity["surface"]:
                    invalid_spans += 1
                if entity["canonical"] not in palette:
                    invalid_palette += 1
                if not any(
                    token["start"] < end and token["end"] > start and tag.endswith("-color")
                    for token, tag in zip(tokens, tags)
                ):
                    invalid_bio += 1
    connection.close()
    if invalid_spans or invalid_bio or invalid_palette:
        raise AssertionError(
            f"invalid_spans={invalid_spans}, invalid_bio={invalid_bio}, invalid_palette={invalid_palette}"
        )
    if expected_metrics["palette_size"] != 30 or expected_metrics["regular_expressions"]:
        raise AssertionError("invalid color normalization contract")
    print("PASSED")
    print(
        f"colors={actual[0]:,}, queries={actual[2]:,}, canonicalized={actual[3]:,}, "
        f"rejected={actual[1]:,}, palette=30"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
