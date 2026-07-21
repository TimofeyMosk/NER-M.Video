#!/usr/bin/env python3
"""Print deterministic entity examples for human inspection."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb


DATASET = Path(__file__).resolve().parent / "output" / "silver_labels.parquet"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    connection = duckdb.connect()
    rows = connection.execute(
        "SELECT annotation_id, query_normalized, category, entities_json "
        "FROM read_parquet(?) WHERE entity_count > 0 ORDER BY MD5(annotation_id || '|entity-audit')",
        [str(DATASET)],
    ).fetchall()
    examples: dict[str, list[tuple[str, str, str, float, str]]] = defaultdict(list)
    confidences: dict[str, list[float]] = defaultdict(list)
    for _, query, category, entities_json in rows:
        for entity in json.loads(entities_json):
            entity_type = entity["type"]
            confidence = float(entity["confidence"])
            confidences[entity_type].append(confidence)
            if len(examples[entity_type]) < 8:
                examples[entity_type].append(
                    (query, category, entity["text"], confidence, entity["value"])
                )
    connection.close()
    for entity_type in sorted(examples):
        values = confidences[entity_type]
        print(f"\n=== {entity_type}: n={len(values)}, min={min(values):.3f}, avg={sum(values)/len(values):.3f} ===")
        for query, category, surface, confidence, value in examples[entity_type]:
            print(f"[{confidence:.3f}] {surface!r} -> {value!r} | {category} | {query}")


if __name__ == "__main__":
    main()
