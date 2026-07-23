#!/usr/bin/env python3
"""Build a catalog-backed color alias CSV for the fixed 30-color palette."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_dictionaries.color_palette import (
    COLOR_NORMALIZATION_VERSION,
    COLOR_SEEDS,
    PALETTE_30,
    PALETTE_BY_ID,
    PHRASE_OVERRIDES,
    normalize_color_text,
    split_color_tokens,
)


DEFAULT_CATALOG = ROOT / "pavel_nntp" / "data" / "processed" / "catalog_attributes.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def russian_forms(seed: str, morph: pymorphy3.MorphAnalyzer) -> set[str]:
    if not seed.isalpha() or not any("а" <= char <= "я" for char in seed):
        return {seed}
    parses = morph.parse(seed)
    if not parses:
        return {seed}
    forms = {normalize_color_text(item.word) for item in parses[0].lexeme}
    forms.add(normalize_color_text(seed))
    return {item for item in forms if item}


def make_manual_aliases() -> dict[str, set[str]]:
    morph = pymorphy3.MorphAnalyzer()
    aliases: dict[str, set[str]] = defaultdict(set)
    for color_id, seeds in COLOR_SEEDS.items():
        for seed in seeds:
            aliases[color_id].update(russian_forms(normalize_color_text(seed), morph))
    for phrase, color_id in PHRASE_OVERRIDES.items():
        aliases[color_id].add(normalize_color_text(phrase))
    for color_id, label_ru, label_en in PALETTE_30:
        aliases[color_id].update({normalize_color_text(label_ru), normalize_color_text(label_en)})
    return dict(aliases)


def classify_value(value: str, token_index: dict[str, set[str]]) -> tuple[str | None, str]:
    normalized = normalize_color_text(value)
    if not normalized:
        return None, "empty"
    if normalized in PHRASE_OVERRIDES:
        return PHRASE_OVERRIDES[normalized], "phrase_override"
    found: set[str] = set()
    for token in split_color_tokens(normalized):
        found.update(token_index.get(token, ()))
    if len(found) == 1:
        return next(iter(found)), "color_anchor"
    if len(found) > 1:
        return "multicolor", "multiple_color_anchors"
    return None, "no_safe_anchor"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manual = make_manual_aliases()
    single_token_index: dict[str, set[str]] = defaultdict(set)
    for color_id, aliases in manual.items():
        for alias in aliases:
            tokens = split_color_tokens(alias)
            if len(tokens) == 1:
                single_token_index[tokens[0]].add(color_id)

    connection = duckdb.connect()
    rows = connection.execute(
        """
        SELECT attribute_value, SUM(n)::BIGINT AS catalog_count
        FROM (
            SELECT attribute_value, count(*) AS n
            FROM read_parquet(?)
            WHERE attribute_value IS NOT NULL
              AND attribute_name IN ('цвет', 'цвет производителя', 'цвет корпуса')
            GROUP BY attribute_value
        )
        GROUP BY attribute_value
        ORDER BY catalog_count DESC, attribute_value
        """,
        [str(args.catalog.resolve())],
    ).fetchall()
    connection.close()

    aggregated: dict[tuple[str, str], dict[str, object]] = {}
    review: list[dict[str, object]] = []
    mapped_occurrences = 0
    reason_counts: Counter[str] = Counter()

    def add_alias(color_id: str, alias: str, source: str, count: int, reason: str) -> None:
        normalized = normalize_color_text(alias)
        if not normalized:
            return
        key = (normalized, color_id)
        row = aggregated.setdefault(key, {
            "color_id": color_id,
            "canonical": PALETTE_BY_ID[color_id][1],
            "surface": alias,
            "normalized": normalized,
            "source": source,
            "catalog_count": 0,
            "mapping_reason": reason,
            "active": "true",
            "version": COLOR_NORMALIZATION_VERSION,
        })
        row["catalog_count"] = int(row["catalog_count"]) + int(count)

    for color_id, aliases in manual.items():
        for alias in aliases:
            add_alias(color_id, alias, "manual_palette", 0, "manual_alias")

    for value, count in rows:
        color_id, reason = classify_value(str(value), single_token_index)
        reason_counts[reason] += 1
        if color_id is None:
            review.append({
                "surface": value,
                "normalized": normalize_color_text(value),
                "catalog_count": int(count),
                "status": "review_only_not_auto_applied",
                "reason": reason,
            })
            continue
        mapped_occurrences += int(count)
        add_alias(color_id, str(value), "catalog_color_attributes", int(count), reason)

    palette_rows = [
        {"rank": index, "color_id": color_id, "canonical": ru, "label_en": en,
         "version": COLOR_NORMALIZATION_VERSION}
        for index, (color_id, ru, en) in enumerate(PALETTE_30, 1)
    ]
    alias_rows = sorted(aggregated.values(), key=lambda row: (
        -int(row["catalog_count"]), str(row["color_id"]), str(row["normalized"])
    ))
    review.sort(key=lambda row: (-int(row["catalog_count"]), str(row["normalized"])))
    total_occurrences = sum(int(count) for _, count in rows)
    metrics = {
        "version": COLOR_NORMALIZATION_VERSION,
        "palette_size": len(PALETTE_30),
        "catalog_distinct_values": len(rows),
        "catalog_occurrences": total_occurrences,
        "mapped_distinct_values": len(rows) - len(review),
        "mapped_occurrences": mapped_occurrences,
        "weighted_coverage_percent": 0.0 if not total_occurrences else 100 * mapped_occurrences / total_occurrences,
        "alias_rows": len(alias_rows),
        "review_rows": len(review),
        "mapping_reason_counts": dict(reason_counts),
        "regular_expressions": False,
        "source_attributes": ["цвет", "цвет производителя", "цвет корпуса"],
    }
    write_csv(output / "color_palette.csv", list(palette_rows[0]), palette_rows)
    write_csv(output / "color_aliases.csv", list(alias_rows[0]), alias_rows)
    write_csv(output / "color_mapping_review.csv", list(review[0]), review)
    (output / "color_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
