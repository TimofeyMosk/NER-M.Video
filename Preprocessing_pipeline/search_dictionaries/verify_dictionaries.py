#!/usr/bin/env python3
"""Validate generated search dictionaries."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_preprocessing.preprocess_queries import (
    NORMALIZATION_STEPS,
    NORMALIZATION_VERSION,
    SEPARATOR_CHARS,
    SafeLemmatizer,
    load_protected_brands,
    normalize_for_lookup,
)
from units import normalize_unit_alias


OUTPUT = Path(__file__).resolve().parent / "output"


def load(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def validate_csv_exports(metrics: dict[str, object]) -> dict[str, object]:
    expected_fields = [
        "entity_type", "entity_id", "canonical", "surface", "normalized",
        "language", "variant", "source", "catalog_count", "active",
        "ambiguity_count", "metadata_json",
    ]
    files = {
        "brands": OUTPUT / "brands.csv",
        "categories": OUTPUT / "categories.csv",
        "units": OUTPUT / "measurement_units.csv",
        "measurements": OUTPUT / "measurement_phrases.csv",
        "combined": OUTPUT / "dictionary.csv",
    }
    expected_counts = metrics["csv_exports"]["rows"]
    actual_counts: dict[str, int] = {}
    language_counts: Counter[tuple[str, str]] = Counter()
    measurement_variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    measurement_units: set[str] = set()
    lossy_active = 0
    sample_rows: list[dict[str, str]] = []

    for name, path in files.items():
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected_fields:
                raise AssertionError(f"unexpected CSV schema in {path.name}: {reader.fieldnames}")
            for row in reader:
                count += 1
                normalized = row["normalized"]
                if normalized != normalized.lower() or "ё" in normalized:
                    raise AssertionError(f"normalization mismatch in {path.name}: {normalized!r}")
                if any(char in SEPARATOR_CHARS for char in normalized):
                    raise AssertionError(f"separator survived normalization in {path.name}: {normalized!r}")
                if normalized != " ".join(normalized.split()):
                    raise AssertionError(f"whitespace is not normalized in {path.name}: {normalized!r}")
                if row["active"] == "true" and not normalized:
                    raise AssertionError(f"active empty surface in {path.name}")
                language_counts[(row["entity_type"], row["language"])] += 1
                if row["entity_type"] == "measurement":
                    measurement_variants[(row["entity_id"], row["language"])].add(row["variant"])
                    measurement_units.add(json.loads(row["metadata_json"])["unit"])
                if row["variant"] == "lossy_separator" and row["active"] == "true":
                    lossy_active += 1
                if name != "combined" and count % 997 == 1:
                    sample_rows.append(row)
        actual_counts[name] = count
        if count != int(expected_counts[name]):
            raise AssertionError(f"CSV row count mismatch for {name}: {count} != {expected_counts[name]}")

    if actual_counts["combined"] != sum(actual_counts[name] for name in files if name != "combined"):
        raise AssertionError("combined dictionary does not equal the sum of entity CSV files")
    if lossy_active:
        raise AssertionError("quote-only unit aliases must be inactive after separator replacement")
    if not all(language_counts[(kind, language)] for kind in ("category", "unit", "measurement") for language in ("ru", "en")):
        raise AssertionError("RU/EN surfaces are missing for categories, units, or measurements")
    paired = sum(variants >= {"compact", "spaced"} for variants in measurement_variants.values())
    if paired == 0 or not {"kg", "inch", "cm"}.issubset(measurement_units):
        raise AssertionError("compact/spaced measurement pairs or core size units are missing")

    lemmatizer = SafeLemmatizer(load_protected_brands())
    for row in sample_rows:
        if row["normalized"] != normalize_for_lookup(row["surface"], lemmatizer):
            raise AssertionError(
                f"CSV normalization differs from preprocessing for {row['entity_type']}: {row['surface']!r}"
            )
    return {
        "rows": actual_counts,
        "ru_en_counts": dict(language_counts),
        "compact_spaced_pairs": paired,
    }


def main() -> int:
    brands = load("brands.json")
    brand_lookup = load("brand_lookup.json")
    categories = load("categories.json")
    category_lookup = load("category_lookup.json")
    units = load("measurement_units.json")
    numbers = load("measurement_numbers.json")
    metrics = load("dictionary_metrics.json")
    contract = load("normalization_contract.json")

    brand_names = {item["canonical"] for item in brands}
    if len(brand_names) != len(brands) or len(brands) != metrics["brands"]["records"]:
        raise AssertionError("brand records are not unique or metrics disagree")
    if any(not set(targets).issubset(brand_names) for targets in brand_lookup.values()):
        raise AssertionError("brand lookup contains an unknown target")
    for surface, canonical in {"apple": "apple", "samsung": "samsung", "xiaomi": "xiaomi", "эпл": "apple"}.items():
        if canonical not in brand_lookup.get(surface, []):
            raise AssertionError(f"required brand surface is absent: {surface} -> {canonical}")

    category_ids = {item["category_id"] for item in categories}
    if len(category_ids) != len(categories) or len(categories) != metrics["categories"]["records"]:
        raise AssertionError("category records are not unique or metrics disagree")
    if any(not set(targets).issubset(category_ids) for targets in category_lookup.values()):
        raise AssertionError("category lookup contains an unknown target")
    if not category_lookup.get("смартфон"):
        raise AssertionError("lemmatized category surface 'смартфон' is absent")

    canonical_units = {item["canonical"] for item in units["units"]}
    if len(canonical_units) != len(units["units"]):
        raise AssertionError("canonical units are not unique")
    if units["alias_collisions"]:
        raise AssertionError(f"unit aliases collide: {units['alias_collisions']}")
    required = {
        "килограммы": "kg",
        "кг": "kg",
        "дюймы": "inch",
        '"': "inch",
        "сантиметры": "cm",
        "см": "cm",
        "миллиметры": "mm",
        "литры": "l",
        "мач": "mah",
    }
    for alias, expected in required.items():
        actual = units["alias_lookup"].get(normalize_unit_alias(alias))
        if actual != expected:
            raise AssertionError(f"unit alias {alias!r}: expected {expected}, got {actual}")
    if not set(numbers).issubset(canonical_units):
        raise AssertionError("measurement_numbers contains an unknown unit")

    connection = duckdb.connect()
    rows, bad_numbers, known_units = connection.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE numbers_json IS NULL OR numbers_json='[]'), "
        "COUNT(DISTINCT canonical_unit) FROM read_parquet(?)",
        [str(OUTPUT / "measurement_values.parquet")],
    ).fetchone()
    connection.close()
    if int(rows) != int(metrics["measurements"]["measurement_value_rows"]):
        raise AssertionError("measurement parquet row count disagrees with metrics")
    if bad_numbers or int(known_units) != int(metrics["measurements"]["canonical_units_with_numbers"]):
        raise AssertionError("measurement parquet contains invalid numbers or unit count")
    if float(metrics["measurements"]["unit_row_coverage"]) < 0.99:
        raise AssertionError("catalog unit coverage is below 99%")

    if contract["version"] != NORMALIZATION_VERSION or tuple(contract["steps"]) != NORMALIZATION_STEPS:
        raise AssertionError("dictionary and query preprocessing contracts differ")
    observed = {item["step"]: item for item in contract["observed_on_preprocessing_corpus"]}
    if not 31.0 <= float(observed["safe_lemmatization"]["changed_share_percent"]) <= 32.0:
        raise AssertionError("lemmatization share no longer matches the audited corpus")
    if not 0.35 <= float(observed["yo_to_e"]["changed_share_percent"]) <= 0.45:
        raise AssertionError("yo-to-e share no longer matches the audited corpus")

    csv_result = validate_csv_exports(metrics)
    with (OUTPUT / "typo_candidates.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        typo_rows = list(csv.DictReader(stream))
    if len(typo_rows) != int(metrics["typo_audit"]["total"]):
        raise AssertionError("typo audit count disagrees with metrics")
    if any(row["action"] != "review_only_not_auto_applied" for row in typo_rows):
        raise AssertionError("possible typos must not be corrected automatically")
    if (OUTPUT / "work").exists():
        raise AssertionError("temporary work directory was not removed")

    print("PASSED")
    print(
        f"brands={len(brands):,}, categories={len(categories):,}, "
        f"units={len(canonical_units):,}, measurement_rows={int(rows):,}, "
        f"unique_numbers={metrics['measurements']['unique_numbers_total']:,}, "
        f"unit_coverage={100 * metrics['measurements']['unit_row_coverage']:.2f}%, "
        f"csv_rows={csv_result['rows']['combined']:,}, "
        f"compact_spaced_pairs={csv_result['compact_spaced_pairs']:,}, "
        f"typo_candidates={len(typo_rows):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
