from __future__ import annotations

import hashlib
import html
import json
import math
import pickle
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "cu_ws"
QUERY_CLICKS_PATH = SOURCE_DIR / "query_clicks.parquet"
CATALOG_PATH = SOURCE_DIR / "skus.pkl"
SKU_DESC_PATH = SOURCE_DIR / "sku_desc.parquet"

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DICTIONARIES_DIR = DATA_DIR / "dictionaries"
CONFIG_DIR = DATA_DIR / "config"
REPORTS_DIR = DATA_DIR / "reports"
WORK_DIR = DATA_DIR / "_work"

CATALOG_SKUS_PATH = PROCESSED_DIR / "catalog_skus.parquet"
CATALOG_ATTRIBUTES_PATH = PROCESSED_DIR / "catalog_attributes.parquet"
QUERY_MAPPING_PATH = WORK_DIR / "query_mapping.parquet"
QUERY_SKU_PATH = PROCESSED_DIR / "query_sku_aggregated.parquet"
QUERY_DATASET_PATH = PROCESSED_DIR / "query_dataset.parquet"
NO_CLICK_PATH = PROCESSED_DIR / "no_click_queries.parquet"
REJECTED_PATH = PROCESSED_DIR / "rejected_rows.parquet"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VALID_PATH = PROCESSED_DIR / "valid.parquet"

REPORT_PATH = REPORTS_DIR / "data_processing_report.html"
CONFIG_PATH = CONFIG_DIR / "processing_config.yaml"
MANIFEST_PATH = DATA_DIR / "manifest.json"
DATA_README_PATH = DATA_DIR / "README.md"

SEED = 42
VALID_FRACTION = 0.20
POSITION_CAP = 100
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
HIGH_CONFIDENCE_MIN_CLICKS = 3
MAX_QUERY_WORDS = 20
MAX_QUERY_CHARS = 200
MIN_STRATUM_SIZE = 5

DASHES_RE = re.compile(
    r"[-\u00ad\u058a\u05be\u1400\u1806\u2010-\u2015\u2212"
    r"\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\u30a0"
    r"\ufe31-\ufe32\ufe58\ufe63\uff0d]"
)
WHITESPACE_RE = re.compile(r"\s+")

BRAND_ALIASES = {
    "самсунг": "samsung",
    "эпл": "apple",
    "сяоми": "xiaomi",
    "ксиаоми": "xiaomi",
    "хуавей": "huawei",
    "хонор": "honor",
    "реалми": "realme",
    "леново": "lenovo",
    "асус": "asus",
    "эйсер": "acer",
    "бош": "bosch",
    "филипс": "philips",
}

PRODUCT_FAMILY_ALIASES = {
    "айфон": "iphone",
    "айпад": "ipad",
    "макбук": "macbook",
    "аирподс": "airpods",
    "эпл вотч": "apple watch",
    "плейстейшен": "playstation",
    "иксбокс": "xbox",
}

PRODUCT_FAMILY_TO_BRAND = {
    "iphone": "apple",
    "ipad": "apple",
    "macbook": "apple",
    "airpods": "apple",
    "apple watch": "apple",
    "playstation": "sony",
    "xbox": "microsoft",
}

ALL_ALIASES = {**BRAND_ALIASES, **PRODUCT_FAMILY_ALIASES}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def prepare_directories() -> None:
    for directory in (PROCESSED_DIR, DICTIONARIES_DIR, CONFIG_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    generated_files = (
        CATALOG_SKUS_PATH,
        CATALOG_ATTRIBUTES_PATH,
        QUERY_SKU_PATH,
        QUERY_DATASET_PATH,
        NO_CLICK_PATH,
        REJECTED_PATH,
        TRAIN_PATH,
        VALID_PATH,
        REPORT_PATH,
        CONFIG_PATH,
        MANIFEST_PATH,
        DATA_README_PATH,
    )
    for path in generated_files:
        path.unlink(missing_ok=True)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = DASHES_RE.sub(" ", text)
    cleaned: list[str] = []
    for char in text:
        if char.isspace():
            cleaned.append(" ")
        elif unicodedata.category(char)[0] in {"L", "N"}:
            cleaned.append(char)
        elif char in "+./":
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()


def make_alias_pattern(aliases: dict[str, str]) -> re.Pattern[str]:
    alternatives = sorted((re.escape(key) for key in aliases), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


ALIAS_PATTERN = make_alias_pattern(ALL_ALIASES)
BRAND_ALIAS_PATTERN = make_alias_pattern(BRAND_ALIASES)
PRODUCT_ALIAS_PATTERN = make_alias_pattern(PRODUCT_FAMILY_ALIASES)


def canonicalize_text(normalized: str) -> str:
    if not normalized:
        return ""
    return ALIAS_PATTERN.sub(lambda match: ALL_ALIASES[match.group(0)], normalized)


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def safe_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dictionary_files(
    categories: list[dict[str, object]],
    brand_counts: Counter[str],
    attribute_counts: Counter[str],
    selected_attribute_values: dict[str, set[str]],
) -> None:
    write_json(DICTIONARIES_DIR / "brand_aliases.json", BRAND_ALIASES)
    write_json(DICTIONARIES_DIR / "product_family_aliases.json", PRODUCT_FAMILY_ALIASES)
    write_json(DICTIONARIES_DIR / "product_family_to_brand.json", PRODUCT_FAMILY_TO_BRAND)
    write_json(DICTIONARIES_DIR / "category_aliases.json", {})
    write_json(DICTIONARIES_DIR / "attribute_aliases.json", {})
    write_json(DICTIONARIES_DIR / "categories.json", categories)
    write_json(DICTIONARIES_DIR / "brands.json", brand_counts.most_common())
    write_json(DICTIONARIES_DIR / "attribute_names.json", attribute_counts.most_common())
    write_json(
        DICTIONARIES_DIR / "selected_attribute_values.json",
        {name: sorted(values) for name, values in sorted(selected_attribute_values.items())},
    )


def extract_catalog() -> dict[str, int]:
    log("Загрузка skus.pkl и построение каталога")
    with CATALOG_PATH.open("rb") as file:
        obj = pickle.load(file)

    shop = obj["yml_catalog"]["shop"]
    raw_categories = shop["categories"]["category"]
    offers = shop["offers"]["offer"]
    offer_count = len(offers)

    category_by_id = {str(item["@id"]): item for item in raw_categories}

    @lru_cache(maxsize=None)
    def category_path(category_id: str) -> tuple[str, ...]:
        item = category_by_id.get(str(category_id))
        if not item:
            return ()
        name = normalize_text(item.get("#text", ""))
        parent_id = item.get("@parentId")
        if parent_id is None or str(parent_id) == str(category_id):
            return (name,) if name else ()
        return (*category_path(str(parent_id)), name) if name else category_path(str(parent_id))

    categories = []
    for item in raw_categories:
        category_id = str(item["@id"])
        categories.append(
            {
                "category_id": category_id,
                "parent_id": str(item["@parentId"]) if item.get("@parentId") is not None else None,
                "name_original": item.get("#text", ""),
                "name_normalized": normalize_text(item.get("#text", "")),
                "path": list(category_path(category_id)),
            }
        )

    catalog_schema = pa.schema(
        [
            ("sku_id", pa.int64()),
            ("catalog_sku_name", pa.string()),
            ("catalog_sku_name_normalized", pa.string()),
            ("catalog_brand_original", pa.string()),
            ("catalog_brand", pa.string()),
            ("catalog_model", pa.string()),
            ("category_id", pa.string()),
            ("category_name", pa.string()),
            ("category_path", pa.string()),
            ("available", pa.bool_()),
        ]
    )
    attribute_schema = pa.schema(
        [
            ("sku_id", pa.int64()),
            ("attribute_name_original", pa.string()),
            ("attribute_name", pa.string()),
            ("attribute_value_original", pa.string()),
            ("attribute_value", pa.string()),
            ("unit", pa.string()),
        ]
    )

    catalog_writer = pq.ParquetWriter(CATALOG_SKUS_PATH, catalog_schema, compression="zstd")
    attribute_writer = pq.ParquetWriter(CATALOG_ATTRIBUTES_PATH, attribute_schema, compression="zstd")
    catalog_buffer: dict[str, list[object]] = defaultdict(list)
    attribute_buffer: dict[str, list[object]] = defaultdict(list)

    brand_counts: Counter[str] = Counter()
    attribute_counts: Counter[str] = Counter()
    selected_attribute_values: dict[str, set[str]] = defaultdict(set)
    invalid_sku_ids = 0
    multi_category_offers = 0
    attribute_rows = 0

    selected_attributes = {
        "страна",
        "форма",
        "цвет",
        "материал",
        "материал корпуса",
    }

    def flush_catalog() -> None:
        if catalog_buffer["sku_id"]:
            catalog_writer.write_table(pa.Table.from_pydict(catalog_buffer, schema=catalog_schema))
            catalog_buffer.clear()

    def flush_attributes() -> None:
        if attribute_buffer["sku_id"]:
            attribute_writer.write_table(pa.Table.from_pydict(attribute_buffer, schema=attribute_schema))
            attribute_buffer.clear()

    for index, offer in enumerate(offers, 1):
        sku_id = safe_int(offer.get("@id"))
        if sku_id is None:
            invalid_sku_ids += 1
            continue

        category_ids = [str(value) for value in as_list((offer.get("categories") or {}).get("categoryId"))]
        if len(category_ids) > 1:
            multi_category_offers += 1
        category_id = category_ids[0] if category_ids else ""
        category_item = category_by_id.get(category_id, {})
        brand_original = str(offer.get("vendor") or "")
        brand = canonicalize_text(normalize_text(brand_original))

        catalog_buffer["sku_id"].append(sku_id)
        catalog_buffer["catalog_sku_name"].append(str(offer.get("name") or ""))
        catalog_buffer["catalog_sku_name_normalized"].append(normalize_text(offer.get("name") or ""))
        catalog_buffer["catalog_brand_original"].append(brand_original)
        catalog_buffer["catalog_brand"].append(brand)
        catalog_buffer["catalog_model"].append(str(offer.get("model") or ""))
        catalog_buffer["category_id"].append(category_id)
        catalog_buffer["category_name"].append(normalize_text(category_item.get("#text", "")))
        catalog_buffer["category_path"].append(" > ".join(category_path(category_id)))
        catalog_buffer["available"].append(str(offer.get("@available", "")).lower() == "true")
        if brand:
            brand_counts[brand] += 1

        for parameter in as_list(offer.get("param")):
            if not isinstance(parameter, dict):
                continue
            name_original = str(parameter.get("@name") or "")
            if not name_original.endswith("_search"):
                continue
            value_original = str(parameter.get("#text") or "")
            name = normalize_text(name_original[:-7])
            value = normalize_text(value_original)
            attribute_counts[name] += 1
            attribute_rows += 1

            attribute_buffer["sku_id"].append(sku_id)
            attribute_buffer["attribute_name_original"].append(name_original)
            attribute_buffer["attribute_name"].append(name)
            attribute_buffer["attribute_value_original"].append(value_original)
            attribute_buffer["attribute_value"].append(value)
            attribute_buffer["unit"].append(str(parameter.get("@unit") or ""))

            if name in selected_attributes and value:
                values = [normalize_text(part) for part in value.split("/")]
                selected_attribute_values[name].update(part for part in values if part)

        if len(catalog_buffer["sku_id"]) >= 50_000:
            flush_catalog()
        if len(attribute_buffer["sku_id"]) >= 100_000:
            flush_attributes()
        if index % 100_000 == 0:
            log(f"Каталог: обработано {index:,}/{len(offers):,} SKU")

    flush_catalog()
    flush_attributes()
    catalog_writer.close()
    attribute_writer.close()

    write_dictionary_files(categories, brand_counts, attribute_counts, selected_attribute_values)
    del obj, offers, raw_categories

    return {
        "catalog_skus": offer_count,
        "categories": len(categories),
        "brands": len(brand_counts),
        "attribute_names": len(attribute_counts),
        "attribute_rows": attribute_rows,
        "invalid_sku_ids": invalid_sku_ids,
        "multi_category_offers": multi_category_offers,
    }


def configure_duckdb() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(WORK_DIR / "pipeline.duckdb"))
    connection.execute("SET threads = 6")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(f"SET temp_directory = '{sql_path(WORK_DIR / 'duckdb_temp')}'")
    connection.execute("SET preserve_insertion_order = false")
    return connection


def create_query_mapping(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    log("Извлечение уникальных исходных запросов")
    unique_path = WORK_DIR / "unique_queries.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT DISTINCT CAST("toValidUTF8(query_text)" AS VARCHAR) AS query_original
            FROM read_parquet('{sql_path(QUERY_CLICKS_PATH)}')
            WHERE "toValidUTF8(query_text)" IS NOT NULL
        ) TO '{sql_path(unique_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    query_schema = pa.schema(
        [
            ("query_original", pa.string()),
            ("query_normalized", pa.string()),
            ("query_canonical", pa.string()),
            ("query_length_chars", pa.int32()),
            ("query_length_words", pa.int16()),
            ("has_digits", pa.bool_()),
            ("has_latin", pa.bool_()),
            ("has_cyrillic", pa.bool_()),
            ("is_mixed_language", pa.bool_()),
            ("has_dictionary_brand", pa.bool_()),
            ("has_dictionary_product_family", pa.bool_()),
        ]
    )
    writer = pq.ParquetWriter(QUERY_MAPPING_PATH, query_schema, compression="zstd")
    parquet_file = pq.ParquetFile(unique_path)
    total = 0
    empty_after_normalization = 0

    for batch in parquet_file.iter_batches(batch_size=100_000):
        originals = batch.column(0).to_pylist()
        normalized = [normalize_text(value) for value in originals]
        canonical = [canonicalize_text(value) for value in normalized]
        rows = {
            "query_original": originals,
            "query_normalized": normalized,
            "query_canonical": canonical,
            "query_length_chars": [len(value) for value in canonical],
            "query_length_words": [len(value.split()) if value else 0 for value in canonical],
            "has_digits": [any(char.isdigit() for char in value) for value in canonical],
            "has_latin": [bool(re.search(r"[a-z]", value)) for value in canonical],
            "has_cyrillic": [bool(re.search(r"[а-я]", value)) for value in canonical],
            "is_mixed_language": [
                bool(re.search(r"[a-z]", value) and re.search(r"[а-я]", value)) for value in canonical
            ],
            "has_dictionary_brand": [bool(BRAND_ALIAS_PATTERN.search(value)) for value in normalized],
            "has_dictionary_product_family": [
                bool(PRODUCT_ALIAS_PATTERN.search(value)) for value in normalized
            ],
        }
        writer.write_table(pa.Table.from_pydict(rows, schema=query_schema))
        total += len(originals)
        empty_after_normalization += sum(not value for value in normalized)
        if total % 500_000 < len(originals):
            log(f"Нормализация: обработано {total:,} уникальных запросов")

    writer.close()
    unique_path.unlink(missing_ok=True)
    return {"unique_queries": total, "empty_after_normalization": empty_after_normalization}


def create_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW raw_clicks AS
        SELECT
            TRY_CAST(sku_id AS BIGINT) AS sku_id,
            CAST("toValidUTF8(sku_name)" AS VARCHAR) AS log_sku_name,
            CAST("toValidUTF8(sku_brand_name)" AS VARCHAR) AS log_brand,
            TRY_CAST(sku_price AS DOUBLE) AS price,
            CAST(sku_subject_id AS VARCHAR) AS sku_subject_id,
            CAST(sku_seo_id AS VARCHAR) AS sku_seo_id,
            CAST("toValidUTF8(query_text)" AS VARCHAR) AS query_original,
            TRY_CAST(sku_position AS INTEGER) AS position
        FROM read_parquet('{sql_path(QUERY_CLICKS_PATH)}');

        CREATE OR REPLACE VIEW query_mapping AS
        SELECT * FROM read_parquet('{sql_path(QUERY_MAPPING_PATH)}');

        CREATE OR REPLACE VIEW catalog_skus AS
        SELECT * FROM read_parquet('{sql_path(CATALOG_SKUS_PATH)}');
        """
    )


def build_query_sku_dataset(connection: duckdb.DuckDBPyConnection) -> None:
    log("Агрегация положительных строк до query_canonical + sku_id")
    connection.execute(
        f"""
        COPY (
            WITH exact_rows AS (
                SELECT
                    mapping.query_canonical,
                    clicks.query_original,
                    clicks.sku_id,
                    clicks.log_sku_name,
                    clicks.log_brand,
                    clicks.price,
                    clicks.sku_subject_id,
                    clicks.sku_seo_id,
                    clicks.position,
                    catalog.catalog_sku_name,
                    catalog.catalog_sku_name_normalized,
                    catalog.catalog_brand_original,
                    catalog.catalog_brand,
                    catalog.catalog_model,
                    catalog.category_id,
                    catalog.category_name,
                    catalog.category_path,
                    COUNT(*)::BIGINT AS exact_row_count
                FROM raw_clicks AS clicks
                JOIN query_mapping AS mapping USING (query_original)
                JOIN catalog_skus AS catalog USING (sku_id)
                WHERE clicks.position > 0
                  AND mapping.query_canonical <> ''
                GROUP BY ALL
            )
            SELECT
                query_canonical,
                sku_id,
                ANY_VALUE(catalog_sku_name) AS sku_name,
                ANY_VALUE(catalog_sku_name_normalized) AS sku_name_normalized,
                ANY_VALUE(catalog_brand_original) AS brand_original,
                ANY_VALUE(catalog_brand) AS brand,
                ANY_VALUE(catalog_model) AS model,
                ANY_VALUE(category_id) AS category_id,
                ANY_VALUE(category_name) AS category_name,
                ANY_VALUE(category_path) AS category_path,
                SUM(exact_row_count)::BIGINT AS click_count,
                COUNT(*)::BIGINT AS exact_group_count,
                MIN(position)::INTEGER AS min_position,
                MEDIAN(position)::DOUBLE AS median_position,
                AVG(position)::DOUBLE AS mean_position,
                SUM(
                    exact_row_count / LOG2(LEAST(position, {POSITION_CAP}) + 2)
                )::DOUBLE AS evidence_score,
                MIN(price)::DOUBLE AS min_price,
                MEDIAN(price)::DOUBLE AS median_price,
                MAX(price)::DOUBLE AS max_price
            FROM exact_rows
            GROUP BY query_canonical, sku_id
        ) TO '{sql_path(QUERY_SKU_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def build_no_click_dataset(connection: duckdb.DuckDBPyConnection) -> None:
    log("Агрегация запросов без клика")
    connection.execute(
        f"""
        COPY (
            WITH usage AS (
                SELECT
                    mapping.query_canonical,
                    mapping.query_normalized,
                    clicks.query_original,
                    COUNT(*) FILTER (WHERE clicks.position = 0)::BIGINT AS no_click_row_count,
                    COUNT(*) FILTER (WHERE clicks.position > 0)::BIGINT AS positive_row_count
                FROM raw_clicks AS clicks
                JOIN query_mapping AS mapping USING (query_original)
                WHERE mapping.query_canonical <> ''
                GROUP BY mapping.query_canonical, mapping.query_normalized, clicks.query_original
            )
            SELECT
                query_canonical,
                LIST_SLICE(LIST(DISTINCT query_original ORDER BY query_original), 1, 5) AS query_original_examples,
                LIST_SLICE(LIST(DISTINCT query_normalized ORDER BY query_normalized), 1, 5) AS query_normalized_variants,
                SUM(no_click_row_count)::BIGINT AS no_click_row_count,
                SUM(positive_row_count)::BIGINT AS positive_row_count,
                (SUM(positive_row_count) > 0) AS has_positive_click
            FROM usage
            WHERE no_click_row_count > 0
            GROUP BY query_canonical
        ) TO '{sql_path(NO_CLICK_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def build_query_dataset(connection: duckdb.DuckDBPyConnection) -> None:
    log("Агрегация до уровня query_canonical и расчет уверенности")
    connection.execute(
        f"""
        COPY (
            WITH pair_data AS (
                SELECT * FROM read_parquet('{sql_path(QUERY_SKU_PATH)}')
            ),
            query_totals AS (
                SELECT
                    query_canonical,
                    SUM(click_count)::BIGINT AS click_count,
                    COUNT(*)::BIGINT AS unique_sku_count,
                    SUM(evidence_score)::DOUBLE AS total_evidence_score,
                    LIST_SLICE(
                        LIST(CAST(sku_id AS VARCHAR) ORDER BY evidence_score DESC, click_count DESC, sku_id),
                        1,
                        10
                    ) AS top_skus
                FROM pair_data
                GROUP BY query_canonical
            ),
            category_scores AS (
                SELECT
                    query_canonical,
                    category_id,
                    ANY_VALUE(category_name) AS category_name,
                    ANY_VALUE(category_path) AS category_path,
                    SUM(evidence_score)::DOUBLE AS category_score
                FROM pair_data
                WHERE category_id <> ''
                GROUP BY query_canonical, category_id
            ),
            dominant_category AS (
                SELECT * EXCLUDE (rank_number)
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY query_canonical
                            ORDER BY category_score DESC, category_id
                        ) AS rank_number
                    FROM category_scores
                )
                WHERE rank_number = 1
            ),
            brand_scores AS (
                SELECT
                    query_canonical,
                    brand,
                    SUM(evidence_score)::DOUBLE AS brand_score
                FROM pair_data
                WHERE brand <> ''
                GROUP BY query_canonical, brand
            ),
            dominant_brand AS (
                SELECT * EXCLUDE (rank_number)
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY query_canonical
                            ORDER BY brand_score DESC, brand
                        ) AS rank_number
                    FROM brand_scores
                )
                WHERE rank_number = 1
            ),
            query_variants AS (
                SELECT
                    query_canonical,
                    LIST_SLICE(LIST(DISTINCT query_original ORDER BY query_original), 1, 5) AS query_original_examples,
                    LIST_SLICE(LIST(DISTINCT query_normalized ORDER BY query_normalized), 1, 5) AS query_normalized_variants,
                    BOOL_OR(has_digits) AS has_digits,
                    BOOL_OR(has_latin) AS has_latin,
                    BOOL_OR(has_cyrillic) AS has_cyrillic,
                    BOOL_OR(is_mixed_language) AS is_mixed_language,
                    BOOL_OR(has_dictionary_brand) AS has_dictionary_brand,
                    BOOL_OR(has_dictionary_product_family) AS has_dictionary_product_family
                FROM query_mapping
                WHERE query_canonical <> ''
                GROUP BY query_canonical
            ),
            no_click AS (
                SELECT query_canonical, no_click_row_count
                FROM read_parquet('{sql_path(NO_CLICK_PATH)}')
            ),
            combined AS (
                SELECT
                    totals.query_canonical,
                    variants.query_original_examples,
                    variants.query_normalized_variants,
                    LENGTH(totals.query_canonical)::INTEGER AS query_length_chars,
                    CASE
                        WHEN totals.query_canonical = '' THEN 0
                        ELSE LENGTH(totals.query_canonical) - LENGTH(REPLACE(totals.query_canonical, ' ', '')) + 1
                    END::INTEGER AS query_length_words,
                    variants.has_digits,
                    variants.has_latin,
                    variants.has_cyrillic,
                    variants.is_mixed_language,
                    variants.has_dictionary_brand,
                    variants.has_dictionary_product_family,
                    totals.click_count,
                    COALESCE(no_click.no_click_row_count, 0)::BIGINT AS no_click_count,
                    totals.unique_sku_count,
                    totals.top_skus,
                    category.category_id AS dominant_category_id,
                    category.category_name AS dominant_category,
                    category.category_path,
                    category.category_score,
                    CASE
                        WHEN totals.total_evidence_score > 0
                        THEN LEAST(
                            1.0,
                            GREATEST(0.0, category.category_score / totals.total_evidence_score)
                        )
                        ELSE 0
                    END::DOUBLE AS category_confidence,
                    brand.brand AS dominant_brand,
                    brand.brand_score,
                    CASE
                        WHEN totals.total_evidence_score > 0
                        THEN LEAST(
                            1.0,
                            GREATEST(0.0, brand.brand_score / totals.total_evidence_score)
                        )
                        ELSE 0
                    END::DOUBLE AS brand_confidence,
                    totals.total_evidence_score
                FROM query_totals AS totals
                JOIN query_variants AS variants USING (query_canonical)
                LEFT JOIN dominant_category AS category USING (query_canonical)
                LEFT JOIN dominant_brand AS brand USING (query_canonical)
                LEFT JOIN no_click USING (query_canonical)
            )
            SELECT
                *,
                CASE
                    WHEN dominant_category_id IS NULL THEN 'low_confidence'
                    WHEN category_confidence >= {HIGH_CONFIDENCE_THRESHOLD}
                         AND click_count >= {HIGH_CONFIDENCE_MIN_CLICKS}
                        THEN 'high_confidence'
                    WHEN category_confidence >= {MEDIUM_CONFIDENCE_THRESHOLD}
                        THEN 'medium_confidence'
                    ELSE 'low_confidence'
                END AS quality,
                (
                    dominant_category_id IS NOT NULL
                    AND category_confidence >= {MEDIUM_CONFIDENCE_THRESHOLD}
                    AND query_length_words BETWEEN 1 AND {MAX_QUERY_WORDS}
                    AND query_length_chars <= {MAX_QUERY_CHARS}
                ) AS is_training_eligible
            FROM combined
        ) TO '{sql_path(QUERY_DATASET_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def build_rejected_dataset(connection: duckdb.DuckDBPyConnection) -> None:
    log("Формирование таблицы исключенных записей")
    connection.execute(
        f"""
        COPY (
            WITH missing_catalog AS (
                SELECT
                    mapping.query_canonical,
                    clicks.sku_id,
                    COUNT(*)::BIGINT AS row_count,
                    'sku_not_found' AS reason
                FROM raw_clicks AS clicks
                JOIN query_mapping AS mapping USING (query_original)
                LEFT JOIN catalog_skus AS catalog USING (sku_id)
                WHERE clicks.position > 0
                  AND mapping.query_canonical <> ''
                  AND catalog.sku_id IS NULL
                GROUP BY mapping.query_canonical, clicks.sku_id
            ),
            empty_queries AS (
                SELECT
                    '' AS query_canonical,
                    NULL::BIGINT AS sku_id,
                    COUNT(*)::BIGINT AS row_count,
                    'empty_after_normalization' AS reason
                FROM raw_clicks AS clicks
                JOIN query_mapping AS mapping USING (query_original)
                WHERE mapping.query_canonical = ''
                HAVING COUNT(*) > 0
            ),
            low_quality AS (
                SELECT
                    query_canonical,
                    NULL::BIGINT AS sku_id,
                    click_count::BIGINT AS row_count,
                    CASE
                        WHEN query_length_words > {MAX_QUERY_WORDS}
                             OR query_length_chars > {MAX_QUERY_CHARS}
                            THEN 'outlier_query_length'
                        WHEN dominant_category_id IS NULL THEN 'category_not_found'
                        ELSE 'ambiguous_category'
                    END AS reason
                FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')
                WHERE NOT is_training_eligible
            )
            SELECT * FROM missing_catalog
            UNION ALL
            SELECT * FROM empty_queries
            UNION ALL
            SELECT * FROM low_quality
        ) TO '{sql_path(REJECTED_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def build_train_valid(connection: duckdb.DuckDBPyConnection) -> None:
    log("Групповое стратифицированное разбиение train/valid")
    split_query = f"""
        WITH eligible AS (
            SELECT
                *,
                CASE
                    WHEN click_count <= 2 THEN 'rare'
                    WHEN click_count <= 20 THEN 'medium'
                    ELSE 'popular'
                END AS frequency_bucket,
                CASE
                    WHEN query_length_words <= 2 THEN 'short'
                    WHEN query_length_words <= 5 THEN 'middle'
                    ELSE 'long'
                END AS length_bucket,
                CASE
                    WHEN has_dictionary_product_family THEN 'product_family'
                    WHEN dominant_brand IS NOT NULL AND dominant_brand <> '' THEN 'category_and_brand'
                    ELSE 'category_only'
                END AS entity_bucket
            FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')
            WHERE is_training_eligible
        ),
        category_prepared AS (
            SELECT
                *,
                CASE
                    WHEN COUNT(*) OVER (PARTITION BY dominant_category_id) < {MIN_STRATUM_SIZE}
                        THEN 'other_rare'
                    ELSE dominant_category_id
                END AS strat_category
            FROM eligible
        ),
        keys AS (
            SELECT
                *,
                CONCAT_WS('|', strat_category, frequency_bucket, length_bucket, entity_bucket, quality) AS key_full,
                CONCAT_WS('|', strat_category, frequency_bucket, length_bucket, entity_bucket) AS key_no_quality,
                CONCAT_WS('|', strat_category, frequency_bucket, length_bucket) AS key_basic,
                CONCAT_WS('|', strat_category, frequency_bucket) AS key_category_frequency,
                strat_category AS key_category
            FROM category_prepared
        ),
        counted AS (
            SELECT
                *,
                COUNT(*) OVER (PARTITION BY key_full) AS n_full,
                COUNT(*) OVER (PARTITION BY key_no_quality) AS n_no_quality,
                COUNT(*) OVER (PARTITION BY key_basic) AS n_basic,
                COUNT(*) OVER (PARTITION BY key_category_frequency) AS n_category_frequency,
                COUNT(*) OVER (PARTITION BY key_category) AS n_category
            FROM keys
        ),
        strata AS (
            SELECT
                *,
                CASE
                    WHEN n_full >= {MIN_STRATUM_SIZE} THEN key_full
                    WHEN n_no_quality >= {MIN_STRATUM_SIZE} THEN key_no_quality
                    WHEN n_basic >= {MIN_STRATUM_SIZE} THEN key_basic
                    WHEN n_category_frequency >= {MIN_STRATUM_SIZE} THEN key_category_frequency
                    WHEN n_category >= {MIN_STRATUM_SIZE} THEN key_category
                    ELSE 'other_rare'
                END AS stratum
            FROM counted
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY stratum
                    ORDER BY MD5(query_canonical || '|{SEED}')
                ) AS stratum_row_number,
                COUNT(*) OVER (PARTITION BY stratum) AS stratum_size
            FROM strata
        )
        SELECT
            * EXCLUDE (
                key_full,
                key_no_quality,
                key_basic,
                key_category_frequency,
                key_category,
                n_full,
                n_no_quality,
                n_basic,
                n_category_frequency,
                n_category,
                stratum_row_number,
                stratum_size,
                strat_category
            ),
            CASE
                WHEN stratum_row_number <= GREATEST(1, FLOOR(stratum_size * {VALID_FRACTION}))
                    THEN 'valid'
                ELSE 'train'
            END AS split
        FROM ranked
    """

    connection.execute(
        f"""
        COPY (SELECT * FROM ({split_query}) WHERE split = 'train')
        TO '{sql_path(TRAIN_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    connection.execute(
        f"""
        COPY (SELECT * FROM ({split_query}) WHERE split = 'valid')
        TO '{sql_path(VALID_PATH)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int | float:
    value = connection.execute(query).fetchone()[0]
    return 0 if value is None else value


def collect_metrics(
    connection: duckdb.DuckDBPyConnection,
    catalog_metrics: dict[str, int],
    query_metrics: dict[str, int],
) -> dict[str, object]:
    log("Проверка результатов и расчет контрольных метрик")
    metrics: dict[str, object] = {**catalog_metrics, **query_metrics}
    metrics.update(
        {
            "raw_rows": scalar(connection, "SELECT COUNT(*) FROM raw_clicks"),
            "positive_rows": scalar(connection, "SELECT COUNT(*) FROM raw_clicks WHERE position > 0"),
            "no_click_rows": scalar(connection, "SELECT COUNT(*) FROM raw_clicks WHERE position = 0"),
            "query_sku_pairs": scalar(
                connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(QUERY_SKU_PATH)}')"
            ),
            "exact_positive_groups": scalar(
                connection,
                f"SELECT SUM(exact_group_count) FROM read_parquet('{sql_path(QUERY_SKU_PATH)}')",
            ),
            "aggregated_queries": scalar(
                connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')"
            ),
            "eligible_queries": scalar(
                connection,
                f"SELECT COUNT(*) FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}') WHERE is_training_eligible",
            ),
            "no_click_queries": scalar(
                connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(NO_CLICK_PATH)}')"
            ),
            "rejected_records": scalar(
                connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(REJECTED_PATH)}')"
            ),
            "train_queries": scalar(connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(TRAIN_PATH)}')"),
            "valid_queries": scalar(connection, f"SELECT COUNT(*) FROM read_parquet('{sql_path(VALID_PATH)}')"),
            "split_overlap": scalar(
                connection,
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_path(TRAIN_PATH)}') AS train
                JOIN read_parquet('{sql_path(VALID_PATH)}') AS valid USING (query_canonical)
                """,
            ),
            "catalog_join_positive_rows": scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM raw_clicks AS clicks
                JOIN catalog_skus AS catalog USING (sku_id)
                WHERE clicks.position > 0
                """,
            ),
        }
    )
    metrics["valid_share"] = (
        metrics["valid_queries"] / (metrics["train_queries"] + metrics["valid_queries"])
        if metrics["train_queries"] + metrics["valid_queries"]
        else 0
    )
    metrics["catalog_join_positive_share"] = (
        metrics["catalog_join_positive_rows"] / metrics["positive_rows"] if metrics["positive_rows"] else 0
    )
    metrics["exact_duplicate_rows_collapsed"] = (
        metrics["catalog_join_positive_rows"] - metrics["exact_positive_groups"]
    )
    if metrics["split_overlap"] != 0:
        raise RuntimeError(f"Обнаружена утечка train/valid: {metrics['split_overlap']} запросов")
    return metrics


def dataframe_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data-table", escape=True)


def build_report(connection: duckdb.DuckDBPyConnection, metrics: dict[str, object]) -> None:
    quality = connection.execute(
        f"""
        SELECT quality AS "Качество", COUNT(*) AS "Запросов"
        FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')
        GROUP BY quality ORDER BY "Запросов" DESC
        """
    ).fetchdf()
    categories = connection.execute(
        f"""
        SELECT dominant_category AS "Категория", COUNT(*) AS "Запросов"
        FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')
        WHERE dominant_category IS NOT NULL
        GROUP BY dominant_category ORDER BY "Запросов" DESC LIMIT 25
        """
    ).fetchdf()
    brands = connection.execute(
        f"""
        SELECT dominant_brand AS "Бренд", COUNT(*) AS "Запросов"
        FROM read_parquet('{sql_path(QUERY_DATASET_PATH)}')
        WHERE dominant_brand IS NOT NULL AND dominant_brand <> ''
        GROUP BY dominant_brand ORDER BY "Запросов" DESC LIMIT 25
        """
    ).fetchdf()
    rejected = connection.execute(
        f"""
        SELECT reason AS "Причина", COUNT(*) AS "Записей", SUM(row_count) AS "Исходных строк"
        FROM read_parquet('{sql_path(REJECTED_PATH)}')
        GROUP BY reason ORDER BY "Исходных строк" DESC
        """
    ).fetchdf()
    split_categories = connection.execute(
        f"""
        WITH combined AS (
            SELECT dominant_category, 'train' AS split FROM read_parquet('{sql_path(TRAIN_PATH)}')
            UNION ALL
            SELECT dominant_category, 'valid' AS split FROM read_parquet('{sql_path(VALID_PATH)}')
        )
        SELECT
            dominant_category AS "Категория",
            COUNT(*) FILTER (WHERE split = 'train') AS "Train",
            COUNT(*) FILTER (WHERE split = 'valid') AS "Valid"
        FROM combined
        GROUP BY dominant_category
        ORDER BY COUNT(*) DESC
        LIMIT 25
        """
    ).fetchdf()

    def metric(label: str, value: object) -> str:
        if isinstance(value, float):
            formatted = f"{value:.1%}"
        else:
            formatted = f"{int(value):,}".replace(",", " ")
        return f"<div class='metric'><strong>{formatted}</strong><span>{html.escape(label)}</span></div>"

    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Отчет по обработке query_clicks</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #20252b; font-family: Inter, system-ui, sans-serif; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 30px 20px 50px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin-top: 30px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: #596572; line-height: 1.5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 22px 0; }}
    .metric {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 13px 15px; }}
    .metric strong {{ display: block; font-size: 23px; }}
    .metric span {{ color: #687480; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .panel {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 14px; overflow: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; }}
    .data-table th, .data-table td {{ padding: 8px 9px; border-bottom: 1px solid #e5e8ec; text-align: left; }}
    .data-table th {{ background: #eef1f4; font-size: 12px; }}
  </style>
</head>
<body><main>
  <h1>Обработка query_clicks</h1>
  <p>Полный проход по логам, объединение с каталогом, агрегация и групповое стратифицированное разбиение. Сформировано {datetime.now().strftime('%Y-%m-%d %H:%M')}.</p>
  <div class="metrics">
    {metric('Исходных строк', metrics['raw_rows'])}
    {metric('Строк с кликом', metrics['positive_rows'])}
    {metric('Строк без клика', metrics['no_click_rows'])}
    {metric('Уникальных исходных запросов', metrics['unique_queries'])}
    {metric('Пар запрос–SKU', metrics['query_sku_pairs'])}
    {metric('Полных дублей схлопнуто', metrics['exact_duplicate_rows_collapsed'])}
    {metric('Агрегированных запросов', metrics['aggregated_queries'])}
    {metric('Train', metrics['train_queries'])}
    {metric('Valid', metrics['valid_queries'])}
    {metric('Доля valid', metrics['valid_share'])}
    {metric('Покрытие каталога', metrics['catalog_join_positive_share'])}
    {metric('Пересечение train/valid', metrics['split_overlap'])}
  </div>
  <div class="grid">
    <section class="panel"><h2>Качество примеров</h2>{dataframe_html(quality)}</section>
    <section class="panel"><h2>Причины исключения</h2>{dataframe_html(rejected)}</section>
    <section class="panel"><h2>Топ категорий</h2>{dataframe_html(categories)}</section>
    <section class="panel"><h2>Топ брендов</h2>{dataframe_html(brands)}</section>
  </div>
  <section class="panel"><h2>Распределение категорий в train/valid</h2>{dataframe_html(split_categories)}</section>
</main></body></html>"""
    REPORT_PATH.write_text(document, encoding="utf-8")


def write_config() -> None:
    aliases = "\n".join(f"    {key!r}: {value!r}" for key, value in sorted(ALL_ALIASES.items()))
    config = f"""version: 1
seed: {SEED}
paths:
  query_clicks: cu_ws/query_clicks.parquet
  catalog: cu_ws/skus.pkl
  sku_desc: cu_ws/sku_desc.parquet
  output: pavel_nntp/data
normalization:
  unicode_form: NFKC
  lowercase: true
  replace_yo_with_e: true
  replace_dashes_with_space: true
  preserve_symbols: ['+', '.', '/']
  fuzzy_mapping: false
aliases:
{aliases}
clicks:
  no_click_position: 0
  positive_position_min: 1
  position_cap: {POSITION_CAP}
  evidence_formula: exact_row_count / log2(min(position, position_cap) + 2)
quality:
  high_confidence_threshold: {HIGH_CONFIDENCE_THRESHOLD}
  high_confidence_min_clicks: {HIGH_CONFIDENCE_MIN_CLICKS}
  medium_confidence_threshold: {MEDIUM_CONFIDENCE_THRESHOLD}
  max_query_words: {MAX_QUERY_WORDS}
  max_query_chars: {MAX_QUERY_CHARS}
split:
  train_fraction: {1 - VALID_FRACTION}
  valid_fraction: {VALID_FRACTION}
  group: query_canonical
  min_stratum_size: {MIN_STRATUM_SIZE}
  stratification: [dominant_category, frequency_bucket, length_bucket, entity_bucket, quality]
"""
    CONFIG_PATH.write_text(config, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(metrics: dict[str, object]) -> None:
    log("Формирование manifest с размерами и контрольными хешами")
    files = []
    for path in sorted(DATA_DIR.rglob("*")):
        if (
            not path.is_file()
            or path.name == ".DS_Store"
            or path == MANIFEST_PATH
            or WORK_DIR in path.parents
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(DATA_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        MANIFEST_PATH,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pipeline": "pavel_nntp/process_query_clicks.py",
            "metrics": metrics,
            "files": files,
        },
    )


def write_data_readme() -> None:
    content = """# Обработанные данные

Папка создана скриптом `pavel_nntp/process_query_clicks.py`.

## processed

- `catalog_skus.parquet` — нормализованный каталог и связь SKU с категориями.
- `catalog_attributes.parquet` — индексируемые `_search` характеристики товаров.
- `query_sku_aggregated.parquet` — агрегированные положительные связи запрос–SKU.
- `query_dataset.parquet` — итоговая таблица на уровне канонического запроса.
- `no_click_queries.parquet` — запросы, для которых встречался `sku_position = 0`.
- `rejected_rows.parquet` — исключенные записи с причиной исключения.
- `train.parquet` — обучающая выборка.
- `valid.parquet` — валидационная выборка.

## dictionaries

Ручные алиасы, категории, бренды и словари характеристик в JSON. Автоматические и fuzzy-алиасы не используются.

## config

`processing_config.yaml` содержит все правила и пороги обработки.

## reports

`data_processing_report.html` содержит контрольные статистики и распределения.

`manifest.json` хранит метрики запуска, размеры файлов и SHA-256.
"""
    DATA_README_PATH.write_text(content, encoding="utf-8")


def validate_sources() -> None:
    missing = [path for path in (QUERY_CLICKS_PATH, CATALOG_PATH, SKU_DESC_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены исходные файлы: " + ", ".join(map(str, missing)))


def main() -> None:
    validate_sources()
    prepare_directories()
    write_config()
    write_data_readme()

    catalog_metrics = extract_catalog()
    connection = configure_duckdb()
    try:
        query_metrics = create_query_mapping(connection)
        create_views(connection)
        build_query_sku_dataset(connection)
        build_no_click_dataset(connection)
        build_query_dataset(connection)
        build_rejected_dataset(connection)
        build_train_valid(connection)
        metrics = collect_metrics(connection, catalog_metrics, query_metrics)
        build_report(connection, metrics)
    finally:
        connection.close()

    write_manifest(metrics)
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    log("Готово. Все результаты сохранены в pavel_nntp/data/")


if __name__ == "__main__":
    main()
