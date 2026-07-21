#!/usr/bin/env python3
"""Build deterministic high-confidence silver labels for search queries.

No regular expressions are used for text matching. Entity spans are produced by
token-sequence matching against catalog-backed candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_preprocessing.preprocess_queries import (
    SafeLemmatizer,
    contains_latin,
    load_protected_brands,
    preprocess,
)


QUERY_DATASET = ROOT / "pavel_nntp" / "data" / "processed" / "query_dataset.parquet"
QUERY_SKU = ROOT / "pavel_nntp" / "data" / "processed" / "query_sku_aggregated.parquet"
ATTRIBUTES = ROOT / "pavel_nntp" / "data" / "processed" / "catalog_attributes.parquet"
PREPROCESSED = ROOT / "text_preprocessing" / "output" / "preprocessed_queries.parquet"
PREPROCESSING_METRICS = ROOT / "text_preprocessing" / "output" / "preprocessing_metrics.json"
BRAND_ALIASES = ROOT / "pavel_nntp" / "data" / "dictionaries" / "brand_aliases.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

ATTRIBUTE_MAP = {
    "цвет": "color",
    "материал": "material",
    "материал корпуса": "material",
    "диагональ": "screen_diagonal",
    "диагональ экрана": "screen_diagonal",
    "диагональ дисплея": "screen_diagonal",
    "частота обновления": "refresh_rate",
    "емкость аккумулятора": "battery_capacity",
    "емкость аккумулятора а ч": "battery_capacity",
    "технология экрана": "display_technology",
    "операционная система": "os",
    "оперативная память ram": "memory_ram",
    "встроенная память rom": "memory_rom",
    "производитель процессора": "processor_brand",
    "модель процессора": "processor_model",
    "потребляемая мощность": "power",
    "максимальная мощность": "power",
    "мощность": "power",
}

ENTITY_PRIORITY = {
    "brand": 110,
    "model": 100,
    "processor_model": 80,
    "processor_brand": 75,
    "memory_ram": 70,
    "memory_rom": 70,
    "screen_diagonal": 65,
    "refresh_rate": 65,
    "battery_capacity": 65,
    "power": 60,
    "display_technology": 55,
    "os": 55,
    "os_version": 55,
    "color": 50,
    "material": 45,
}

GENERIC_MODEL_WORDS = frozenset({"pro", "max", "plus", "ultra", "mini", "air", "new"})
UNIT_ALIASES = {"\"": "дюйм", "″": "дюйм"}


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    entity_type: str
    value: str
    surface: str
    confidence: float
    source: str
    unit: str = ""


def token_spans(text: str, lemmatizer: SafeLemmatizer) -> list[Token]:
    result: list[Token] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isalnum():
            if start is None:
                start = index
        elif start is not None:
            value = text[start:index]
            result.append(Token(value, lemmatizer.lemmatize_token(value), start, index))
            start = None
    if start is not None:
        value = text[start:]
        result.append(Token(value, lemmatizer.lemmatize_token(value), start, len(text)))
    return result


def normalized_surface(text: str, lemmatizer: SafeLemmatizer) -> tuple[str, list[str]]:
    stages = preprocess(str(text), lemmatizer)
    tokens = token_spans(stages.spaces, lemmatizer)
    return stages.spaces, [token.lemma for token in tokens]


def find_sequences(query_tokens: list[Token], candidate_lemmas: list[str]) -> list[tuple[int, int]]:
    if not candidate_lemmas or len(candidate_lemmas) > len(query_tokens):
        return []
    found: list[tuple[int, int]] = []
    width = len(candidate_lemmas)
    for index in range(len(query_tokens) - width + 1):
        if [token.lemma for token in query_tokens[index : index + width]] == candidate_lemmas:
            found.append((index, index + width))
    return found


def contains_sequence(values: list[str], candidate: list[str]) -> bool:
    if not candidate or len(candidate) > len(values):
        return False
    width = len(candidate)
    return any(values[index:index + width] == candidate for index in range(len(values) - width + 1))


def split_attribute_values(value: str) -> list[str]:
    parts = [part.strip() for part in str(value).split("/")]
    return [part for part in parts if part]


def has_letter(text: str) -> bool:
    return any(char.isalpha() for char in text)


def has_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def prepare_candidate_surfaces(
    candidate: Candidate,
    lemmatizer: SafeLemmatizer,
) -> list[tuple[str, list[str], str]]:
    raw_surfaces = [candidate.surface]
    if candidate.entity_type not in {"brand", "model"}:
        raw_surfaces = split_attribute_values(candidate.surface)
    prepared: list[tuple[str, list[str], str]] = []
    for raw in raw_surfaces:
        surface, lemmas = normalized_surface(raw, lemmatizer)
        if not lemmas:
            continue
        if candidate.entity_type == "model":
            joined = " ".join(lemmas)
            if joined in GENERIC_MODEL_WORDS or (len(joined) < 3 and not has_digit(joined)):
                continue
            compact_latin_name = contains_latin(joined) and candidate.confidence >= 0.90 and len(lemmas) <= 3
            if not has_digit(joined) and not compact_latin_name:
                continue
            if len(lemmas) > 5 or len(surface) > 60:
                continue
        if not has_letter(surface):
            if not candidate.unit:
                continue
            base_width = len(lemmas)
            unit = UNIT_ALIASES.get(candidate.unit.strip(), candidate.unit)
            surface, lemmas = normalized_surface(f"{surface} {unit}", lemmatizer)
            added = lemmas[base_width:]
            if not added or not any(has_letter(token) for token in added):
                continue
        matched_value = candidate.value if candidate.entity_type in {"brand", "model"} else surface
        prepared.append((surface, lemmas, matched_value))
    return prepared


def resolve_entities(
    query_text: str,
    candidates: Iterable[Candidate],
    lemmatizer: SafeLemmatizer,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], list[bool], int]:
    tokens = token_spans(query_text, lemmatizer)
    matches: list[dict[str, object]] = []
    for candidate in candidates:
        for candidate_surface, candidate_lemmas, matched_value in prepare_candidate_surfaces(candidate, lemmatizer):
            for token_start, token_end in find_sequences(tokens, candidate_lemmas):
                char_start = tokens[token_start].start
                char_end = tokens[token_end - 1].end
                matches.append({
                    "type": candidate.entity_type,
                    "value": matched_value,
                    "text": query_text[char_start:char_end],
                    "start": char_start,
                    "end": char_end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "confidence": round(float(candidate.confidence), 6),
                    "source": candidate.source,
                    "matched_candidate": candidate_surface,
                    "unit": candidate.unit,
                })

    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for match in matches:
        key = (match["type"], match["value"], match["token_start"], match["token_end"])
        previous = unique.get(key)
        if previous is None or float(match["confidence"]) > float(previous["confidence"]):
            unique[key] = match

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            -ENTITY_PRIORITY.get(str(item["type"]), 0),
            -(int(item["token_end"]) - int(item["token_start"])),
            -float(item["confidence"]),
            int(item["token_start"]),
        ),
    )
    selected: list[dict[str, object]] = []
    occupied: set[int] = set()
    overlap_rejections = 0
    for match in ordered:
        positions = set(range(int(match["token_start"]), int(match["token_end"])))
        if positions & occupied:
            overlap_rejections += 1
            continue
        occupied.update(positions)
        selected.append(match)
    selected.sort(key=lambda item: (int(item["token_start"]), int(item["token_end"])))

    tags = ["O"] * len(tokens)
    supervision_mask = [False] * len(tokens)
    for entity in selected:
        start, end = int(entity["token_start"]), int(entity["token_end"])
        for index in range(start, end):
            prefix = "B" if index == start else "I"
            tags[index] = f"{prefix}-{entity['type']}"
            supervision_mask[index] = True
    token_json = [
        {"text": token.text, "lemma": token.lemma, "start": token.start, "end": token.end}
        for token in tokens
    ]
    return selected, token_json, tags, supervision_mask, overlap_rejections


def load_brand_aliases() -> dict[str, list[str]]:
    aliases = json.loads(BRAND_ALIASES.read_text(encoding="utf-8"))
    by_brand: dict[str, list[str]] = defaultdict(list)
    for alias, brand in aliases.items():
        by_brand[str(brand)].append(str(alias))
    return by_brand


def build_pool(
    connection: duckdb.DuckDBPyConnection,
    work_dir: Path,
    lemmatizer: SafeLemmatizer,
    target: int,
    seed: int,
    shards: int,
) -> None:
    pool_tsv = work_dir / "candidate_pool.tmp"
    cursor = connection.execute(
        "SELECT query_canonical, query_original_examples[1] AS query_original, query_length_words, "
        "has_digits, has_latin, has_cyrillic, is_mixed_language, click_count, no_click_count, "
        "unique_sku_count, dominant_category_id, dominant_category, category_path, category_confidence, "
        "dominant_brand, brand_confidence, "
        "CASE WHEN category_confidence >= 0.90 AND click_count >= 5 THEN 0 ELSE 1 END AS eligibility_tier, "
        "MD5(query_canonical || ?) AS sample_hash "
        "FROM read_parquet(?) WHERE category_confidence >= 0.80 AND click_count >= 3 "
        "AND dominant_category_id IS NOT NULL AND query_length_words BETWEEN 1 AND 20 "
        "AND query_length_chars <= 200 ORDER BY sample_hash",
        [f"|{seed}", str(QUERY_DATASET)],
    )
    fields = [
        "query_canonical", "query_original", "query_normalized", "query_preprocessed",
        "query_length_words", "has_digits", "has_latin", "has_cyrillic", "is_mixed_language",
        "click_count", "no_click_count", "unique_sku_count", "category_id", "category",
        "category_path", "category_confidence", "dominant_brand", "brand_confidence", "eligibility_tier", "sample_hash",
    ]
    with pool_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        while rows := cursor.fetchmany(5000):
            for row in rows:
                original = row[1] or row[0]
                stages = preprocess(original, lemmatizer)
                writer.writerow([
                    row[0], original, stages.spaces, stages.lemma,
                    *row[2:],
                ])

    connection.execute(
        f"CREATE OR REPLACE TABLE candidate_pool AS SELECT * FROM read_csv({sql_string(pool_tsv)}, "
        "delim='\\t', header=true, all_varchar=true, quote='\"', "
        "nullstr='__SILVER_NULL_SENTINEL__')"
    )
    connection.execute(
        f"CREATE OR REPLACE TABLE selected AS "
        "WITH consistent_tiers AS ("
        "  SELECT query_preprocessed, TRY_CAST(eligibility_tier AS INTEGER) AS eligibility_tier "
        "  FROM candidate_pool WHERE query_preprocessed <> '' "
        "  GROUP BY query_preprocessed, TRY_CAST(eligibility_tier AS INTEGER) "
        "  HAVING COUNT(DISTINCT category_id) = 1"
        "), best_tier AS ("
        "  SELECT query_preprocessed, MIN(eligibility_tier) AS eligibility_tier "
        "  FROM consistent_tiers GROUP BY query_preprocessed"
        "), ranked AS ("
        "  SELECT pool.*, ROW_NUMBER() OVER (PARTITION BY pool.query_preprocessed "
        "    ORDER BY TRY_CAST(pool.category_confidence AS DOUBLE) DESC, "
        "    TRY_CAST(pool.click_count AS BIGINT) DESC, pool.sample_hash) AS duplicate_rank "
        "  FROM candidate_pool AS pool JOIN best_tier AS consistent "
        "    ON pool.query_preprocessed = consistent.query_preprocessed "
        "    AND TRY_CAST(pool.eligibility_tier AS INTEGER) = consistent.eligibility_tier "
        f"  JOIN read_parquet({sql_string(PREPROCESSED)}) AS final "
        "    ON final.query_preprocessed = pool.query_preprocessed"
        "), deduplicated AS ("
        "  SELECT * EXCLUDE (duplicate_rank) FROM ranked WHERE duplicate_rank = 1 "
        f"  ORDER BY TRY_CAST(eligibility_tier AS INTEGER), sample_hash LIMIT {target}"
        "), numbered AS ("
        "  SELECT *, ROW_NUMBER() OVER (ORDER BY sample_hash) AS sample_index FROM deduplicated"
        ") SELECT *, ((sample_index - 1) % " + str(shards) + ")::INTEGER AS shard FROM numbered"
    )
    actual = connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    if actual != target:
        raise RuntimeError(f"Недостаточно согласованных кандидатов: target={target}, actual={actual}")
    pool_tsv.unlink(missing_ok=True)


def build_candidate_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"CREATE OR REPLACE TABLE model_candidates AS "
        "WITH sku_ranked AS ("
        " SELECT q.*, ROW_NUMBER() OVER (PARTITION BY q.query_canonical ORDER BY q.evidence_score DESC) AS sku_rank "
        f" FROM read_parquet({sql_string(QUERY_SKU)}) q JOIN selected s USING (query_canonical)"
        "), scores AS ("
        " SELECT query_canonical, model, SUM(evidence_score)::DOUBLE AS score "
        " FROM sku_ranked WHERE sku_rank <= 10 AND model IS NOT NULL AND TRIM(model) <> '' "
        " GROUP BY query_canonical, model"
        "), ranked AS ("
        " SELECT *, score / NULLIF(SUM(score) OVER (PARTITION BY query_canonical), 0) AS confidence, "
        " ROW_NUMBER() OVER (PARTITION BY query_canonical ORDER BY score DESC, model) AS candidate_rank "
        " FROM scores"
        ") SELECT * FROM ranked WHERE candidate_rank <= 10"
    )
    connection.execute("CREATE OR REPLACE TABLE attribute_map(attribute_name VARCHAR, entity_type VARCHAR)")
    connection.executemany("INSERT INTO attribute_map VALUES (?, ?)", list(ATTRIBUTE_MAP.items()))
    connection.execute(
        f"CREATE OR REPLACE TABLE attribute_candidates AS "
        "WITH sku_ranked AS ("
        " SELECT q.query_canonical, q.sku_id, q.evidence_score, "
        " ROW_NUMBER() OVER (PARTITION BY q.query_canonical ORDER BY q.evidence_score DESC) AS sku_rank "
        f" FROM read_parquet({sql_string(QUERY_SKU)}) q JOIN selected s USING (query_canonical)"
        "), scores AS ("
        " SELECT sku.query_canonical, mapping.entity_type, attr.attribute_value, attr.unit, "
        " SUM(sku.evidence_score)::DOUBLE AS score "
        " FROM sku_ranked sku "
        f" JOIN read_parquet({sql_string(ATTRIBUTES)}) attr USING (sku_id) "
        " JOIN attribute_map mapping USING (attribute_name) "
        " WHERE sku.sku_rank <= 10 AND attr.attribute_value IS NOT NULL AND TRIM(attr.attribute_value) <> '' "
        " GROUP BY sku.query_canonical, mapping.entity_type, attr.attribute_value, attr.unit"
        "), ranked AS ("
        " SELECT *, score / NULLIF(SUM(score) OVER (PARTITION BY query_canonical, entity_type), 0) AS confidence, "
        " ROW_NUMBER() OVER (PARTITION BY query_canonical, entity_type ORDER BY score DESC, attribute_value) AS candidate_rank "
        " FROM scores"
        ") SELECT * FROM ranked WHERE candidate_rank <= 10 AND confidence >= 0.80"
    )


def boolean(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _write_legacy_report(metrics: dict[str, object], path: Path) -> None:
    entity_rows = "\n".join(
        f"| `{name}` | {count:,} | {100 * count / metrics['rows']:.2f}% |"
        for name, count in metrics["entity_type_counts"].items()
    ) or "| — | 0 | 0% |"
    category_rows = "\n".join(
        f"| {name} | {count:,} |" for name, count in metrics["top_categories"].items()
    )
    report = f"""# Отчёт о silver-разметке 10% корпуса

Дата: {metrics['generated_at']}.

## Результат

Размечено **{metrics['rows']:,}** уникальных предобработанных запросов — **{metrics['share_of_preprocessed']:.2f}%** от корпуса размером {metrics['preprocessed_corpus_size']:,} строк. Датасет разбит на {metrics['shards']} детерминированных частей и train/valid без пересечения по `query_preprocessed`.

Это высокоуверенная **silver-разметка**, а не ручной gold. Её можно использовать для предварительного обучения и сравнения подходов, но итоговую оценку модели необходимо проводить на независимой ручной выборке.

## Критерии включения

- уровни A/B: `category_confidence >= 0.90` и минимум 5 положительных кликов;
- дополняющий уровень C: `category_confidence >= 0.80` и минимум 3 положительных клика;
- длина 1–20 слов и не более 200 символов;
- категория определена;
- после лемматизации запрос не пуст;
- при схлопывании в одну лемму все варианты согласны по категории;
- итоговый текст существует в полном предобработанном корпусе;
- выборка детерминирована seed={metrics['seed']}.

Уровень `A` означает `category_confidence >= 0.95`, уровень `B` — 0.90–0.95.

## Состав

- Grade A: {metrics['quality_grades'].get('A', 0):,};
- Grade B: {metrics['quality_grades'].get('B', 0):,};
- Grade C: {metrics['quality_grades'].get('C', 0):,};
- train: {metrics['splits'].get('train', 0):,};
- valid: {metrics['splits'].get('valid', 0):,};
- запросов хотя бы с одной явной сущностью: {metrics['rows_with_entities']:,};
- запросов только с меткой категории: {metrics['rows_without_entities']:,};
- всего явных spans: {metrics['entity_total']:,};
- конфликтующих пересечений, разрешённых приоритетами: {metrics['overlap_rejections']:,}.

## Явные сущности

| Тип | Число spans | Доля запросов |
|---|---:|---:|
{entity_rows}

Brand, model и характеристики добавляются только при токенном совпадении с текстом. Для model и характеристик дополнительно требуется подтверждение значением у top-10 кликнутых SKU; значения с confidence ниже 0.80 не рассматриваются. Чисто числовое значение размечается только вместе с единицей измерения. Model должен содержать цифру либо быть компактным латинским названием с confidence не ниже 0.90.

## Частые категории

| Категория | Запросов |
|---|---:|
{category_rows}

## Формат строки

- `query_original` — представитель исходных вариантов;
- `query_normalized` — технически нормализованный текст до лемматизации;
- `query_preprocessed` — итоговый лемматизированный текст;
- `category_*` — полная silver-метка категории и confidence;
- `entities_json` — найденные явные spans с offsets, canonical value, confidence и источником;
- `tokens_json` — токены и offsets;
- `bio_tags_weak_json` — BIO-теги;
- `bio_supervision_mask_json` — `true` только для подтверждённых положительных токенов;
- `split`, `shard`, `quality_grade` — воспроизводимое разбиение и качество.

## Почему entity-разметка частичная

Отсутствие словарного совпадения не доказывает отсутствие сущности. Поэтому токены вне подтверждённых spans получают тег `O`, но `bio_supervision_mask=false`. Для обычного CRF нельзя бездумно превратить весь такой текст в полностью размеченную BIO-последовательность: это создаст ложные отрицательные примеры, особенно для моделей, серий и редких характеристик.

Варианты использования:

1. классификатор категории обучается на всех строках;
2. entity-модель использует positive spans и masked/partial loss;
3. для обычного CRF сначала вручную проверяется подвыборка или строится отдельный fully-labelled subset;
4. `manual_review_sample.csv` используется для оценки precision каждого источника.

## Контроль качества

- уникальность `query_preprocessed`: {metrics['unique_queries_verified']};
- пересечение train/valid: {metrics['split_overlap']};
- spans с некорректными offsets: {metrics['invalid_spans']};
- пустые запросы: {metrics['empty_queries']};
- арифметическая сверка shards: {metrics['shard_reconciliation']}.

## Ограничения и риски

1. Категория наследует click bias и ошибки каталога.
2. Исходный pipeline считает `sku_position=0` отсутствием клика; семантика этого поля всё ещё должна быть подтверждена владельцем данных.
3. Представитель группы запросов не покрывает все поверхностные варианты одной `query_canonical`.
4. Long-tail сущности имеют низкое покрытие; высокая precision получена ценой recall.
5. Series не размечается автоматически: надёжного отдельного поля серии в каталоге нет.
6. Автоматическая выборка не заменяет 3000–5000 ручных gold-запросов.

## Следующий шаг

Вручную проверить минимум 1000 строк из `manual_review_sample.csv`, посчитать precision по источникам `brand_exact`, `model_catalog_exact` и `attribute_catalog_exact`, затем скорректировать пороги. После этого можно обучать category baseline на полном наборе и entity baseline на проверенном/маскированном поднаборе.
"""
    path.write_text(report, encoding="utf-8")


def _write_report_encoded(metrics: dict[str, object], path: Path) -> None:
    """Write a UTF-8 report for the validation-only silver subset."""
    entity_rows = "\n".join(
        f"| `{name}` | {count:,} | {100 * count / metrics['rows']:.2f}% |"
        for name, count in metrics["entity_type_counts"].items()
    ) or "| — | 0 | 0% |"
    category_rows = "\n".join(
        f"| {name} | {count:,} |" for name, count in metrics["top_categories"].items()
    )
    report = f"""# Отчёт о silver-разметке 15% корпуса

Дата: {metrics['generated_at']}.

## Результат

Размечено **{metrics['rows']:,}** уникальных предобработанных запросов — **{metrics['share_of_preprocessed']:.2f}%** от корпуса размером {metrics['preprocessed_corpus_size']:,} строк. Все строки имеют `split=validation`; набор разбит на {metrics['shards']} детерминированных частей только для удобства хранения и обработки.

Первые 10% прежней выборки сохранены благодаря тому же seed и детерминированному порядку, дополнительно включены следующие 5% корпуса.

Это высокоуверенная **silver-разметка**, а не независимый ручной gold. Набор пригоден как рабочая validation-выборка для итераций, но метрики на нём могут быть оптимистичны из-за click bias, ошибок каталога и совпадения механизма разметки с правилами модели. Финальное сравнение моделей следует подтвердить на отдельной ручной gold-подвыборке.

## Критерии включения

- `category_confidence >= 0.90`;
- минимум 5 положительных кликов;
- длина 1–20 слов и не более 200 символов;
- категория определена;
- после лемматизации запрос не пуст;
- варианты, схлопнувшиеся в одну лемму, согласованы по категории;
- итоговый текст присутствует в полном предобработанном корпусе;
- выборка детерминирована seed={metrics['seed']}.

Уровень `A` означает `category_confidence >= 0.95`, уровень `B` — 0.90–0.95.

## Состав

- Validation: {metrics['splits'].get('validation', 0):,};
- Grade A: {metrics['quality_grades'].get('A', 0):,};
- Grade B: {metrics['quality_grades'].get('B', 0):,};
- запросов хотя бы с одной явной сущностью: {metrics['rows_with_entities']:,};
- запросов только с меткой категории: {metrics['rows_without_entities']:,};
- всего явных spans: {metrics['entity_total']:,};
- конфликтующих пересечений, разрешённых приоритетами: {metrics['overlap_rejections']:,}.

## Явные сущности

| Тип | Число spans | Доля запросов |
|---|---:|---:|
{entity_rows}

Brand, model и характеристики добавляются только при точном токенном совпадении с текстом. Для model и характеристик требуется подтверждение значением у top-10 кликнутых SKU; значения с confidence ниже 0.80 не используются. Чисто числовое значение размечается только вместе с единицей измерения.

## Частые категории

| Категория | Запросов |
|---|---:|
{category_rows}

## Формат строки

- `query_original` — представитель исходных вариантов;
- `query_normalized` — нормализованный текст до лемматизации;
- `query_preprocessed` — итоговый лемматизированный текст;
- `category_*` — полная silver-метка категории и confidence;
- `entities_json` — подтверждённые spans с offsets, canonical value, confidence и источником;
- `tokens_json`, `bio_tags_weak_json` — токены и слабые BIO-метки;
- `bio_supervision_mask_json` — `true` только для подтверждённых положительных токенов;
- `split=validation`, `shard`, `quality_grade` — роль, часть и уровень качества.

## Почему entity-разметка частичная

Отсутствие словарного совпадения не доказывает отсутствие сущности. Поэтому токены вне подтверждённых spans получают тег `O`, но `bio_supervision_mask=false`. Нельзя считать такую последовательность полностью размеченной BIO-разметкой: это создаст ложные отрицательные примеры, особенно для моделей, серий и редких характеристик.

## Контроль качества

- уникальность `query_preprocessed`: {metrics['unique_queries_verified']};
- строк не с `split=validation`: {metrics['non_validation_rows']};
- spans с некорректными offsets: {metrics['invalid_spans']};
- пустые запросы: {metrics['empty_queries']};
- арифметическая сверка shards: {metrics['shard_reconciliation']}.

## Ограничения и риски

1. Категория наследует click bias и ошибки каталога.
2. Семантика `sku_position=0` в исходном pipeline требует подтверждения владельцем данных.
3. Entity-метки являются positive-only partial annotations, поэтому обычные полнота, F1 и exact match на всём наборе некорректны без ручного дополнения пропущенных сущностей.
4. Long-tail сущности имеют низкое покрытие; высокая precision получена ценой recall.
5. `series` автоматически не размечается: надёжного отдельного поля серии в каталоге нет.
6. Нельзя обучать модель на этих же 15% и затем называть метрики на них валидационными.

## Следующий шаг

Вручную проверить `manual_review_sample.csv`, дополнить отсутствующие сущности и посчитать precision/recall по типам. Для финального benchmark рекомендуется выделить из этих 15% неизменяемый ручной gold-test, а остальные silver-строки использовать как рабочую validation-выборку.
"""
    path.write_text(report, encoding="utf-8")


def write_report(metrics: dict[str, object], path: Path) -> None:
    """Render the report and append explicit validation quality tiers."""
    _write_report_encoded(metrics, path)
    report = path.read_text(encoding="utf-8")
    report = report.replace(
        "- `category_confidence >= 0.90`;\n- минимум 5 положительных кликов;",
        "- уровни A/B: `category_confidence >= 0.90` и минимум 5 положительных кликов;\n"
        "- дополняющий уровень C: `category_confidence >= 0.80` и минимум 3 положительных клика;",
    )
    report = report.replace(
        f"- Grade B: {metrics['quality_grades'].get('B', 0):,};",
        f"- Grade B: {metrics['quality_grades'].get('B', 0):,};\n"
        f"- Grade C: {metrics['quality_grades'].get('C', 0):,};",
    )
    report = report.replace(
        "Уровень `A` означает `category_confidence >= 0.95`, уровень `B` — 0.90–0.95.",
        "Уровни A/B дополнительно требуют минимум 5 кликов; остальные допустимые строки относятся к C.",
    )
    report += f"""

## Уточнение по уровням качества

- Grade A: `category_confidence >= 0.95`, минимум 5 кликов — {metrics['quality_grades'].get('A', 0):,} строк;
- Grade B: `category_confidence >= 0.90`, минимум 5 кликов — {metrics['quality_grades'].get('B', 0):,} строк;
- Grade C: дополняющий слой с `category_confidence >= 0.80` и минимум 3 кликами — {metrics['quality_grades'].get('C', 0):,} строк.

Сначала в выборку включаются все доступные строки A/B, затем недостающий до 15% объём добирается уровнем C. Уровень нужно учитывать при расчёте метрик и анализировать отдельно.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent", type=float, default=0.15)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output = args.output_dir.resolve()
    shards_dir = output / "shards"
    work_dir = output / "work"
    output.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    pre_metrics = json.loads(PREPROCESSING_METRICS.read_text(encoding="utf-8"))
    corpus_size = int(pre_metrics["counts"]["lemma_unique"])
    target = round(corpus_size * args.percent)
    connection = duckdb.connect(str(work_dir / "annotation.duckdb"))
    connection.execute("SET threads=6")
    connection.execute("SET memory_limit='8GB'")
    lemmatizer = SafeLemmatizer(load_protected_brands())
    brand_aliases = load_brand_aliases()

    print(f"target={target:,}", flush=True)
    build_pool(connection, work_dir, lemmatizer, target, args.seed, args.shards)
    print("selected pool ready", flush=True)
    build_candidate_tables(connection)
    print("catalog candidates ready", flush=True)

    counters = Counter()
    category_counts: Counter[str] = Counter()
    entity_type_counts: Counter[str] = Counter()
    shard_paths: list[Path] = []
    output_fields = [
        "annotation_id", "query_original", "query_normalized", "query_preprocessed",
        "category_id", "category", "category_path", "category_confidence", "click_count",
        "unique_sku_count", "quality_grade", "entities_json", "entity_types_json", "entity_count",
        "tokens_json", "bio_tags_weak_json", "bio_supervision_mask_json", "entity_labels_complete",
        "category_label_complete", "label_source", "split", "shard", "has_digits", "has_latin",
        "has_cyrillic", "is_mixed_language",
    ]

    for shard in range(args.shards):
        print(f"annotating shard {shard + 1}/{args.shards}", flush=True)
        base_rows = connection.execute(
            "SELECT query_canonical, query_original, query_normalized, query_preprocessed, "
            "category_id, category, category_path, TRY_CAST(category_confidence AS DOUBLE), "
            "TRY_CAST(click_count AS BIGINT), TRY_CAST(unique_sku_count AS BIGINT), dominant_brand, "
            "TRY_CAST(brand_confidence AS DOUBLE), has_digits, has_latin, has_cyrillic, is_mixed_language "
            "FROM selected WHERE shard=? ORDER BY sample_index",
            [shard],
        ).fetchall()
        models: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for query, model, confidence in connection.execute(
            "SELECT model.query_canonical, model.model, model.confidence FROM model_candidates model "
            "JOIN selected s USING (query_canonical) WHERE s.shard=? ORDER BY model.query_canonical, model.candidate_rank",
            [shard],
        ).fetchall():
            models[query].append((model, float(confidence or 0)))
        attributes: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
        for query, entity_type, value, unit, confidence in connection.execute(
            "SELECT attr.query_canonical, attr.entity_type, attr.attribute_value, attr.unit, attr.confidence "
            "FROM attribute_candidates attr JOIN selected s USING (query_canonical) "
            "WHERE s.shard=? ORDER BY attr.query_canonical, attr.entity_type, attr.candidate_rank",
            [shard],
        ).fetchall():
            attributes[query].append((entity_type, value, unit or "", float(confidence or 0)))

        temp_path = work_dir / f"part-{shard:03d}.tmp"
        parquet_path = shards_dir / f"part-{shard:03d}.parquet"
        shard_paths.append(parquet_path)
        with temp_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(output_fields)
            for row in base_rows:
                (
                    query_canonical, query_original, query_normalized, query_preprocessed,
                    category_id, category, category_path, category_confidence, click_count,
                    unique_sku_count, dominant_brand, brand_confidence,
                    has_digits_value, has_latin_value, has_cyrillic_value, mixed_value,
                ) = row
                candidates: list[Candidate] = []
                if dominant_brand and brand_confidence >= 0.90:
                    candidates.append(Candidate("brand", dominant_brand, dominant_brand, brand_confidence, "brand_exact"))
                    for alias in brand_aliases.get(dominant_brand, []):
                        candidates.append(Candidate("brand", dominant_brand, alias, brand_confidence, "brand_alias_exact"))
                for model, confidence in models.get(query_canonical, []):
                    if confidence >= 0.80:
                        model_surface = preprocess(model, lemmatizer).spaces
                        brand_surface = preprocess(dominant_brand or "", lemmatizer).spaces
                        if brand_surface and model_surface.startswith(brand_surface + " "):
                            model_surface = model_surface[len(brand_surface):].strip()
                        candidates.append(Candidate("model", model, model_surface, confidence, "model_catalog_exact"))
                _, category_lemmas = normalized_surface(category or "", lemmatizer)
                for entity_type, value, unit, confidence in attributes.get(query_canonical, []):
                    if entity_type == "os":
                        entity_type = "os_version" if has_digit(value) else "os"
                    if entity_type == "material":
                        material_parts = split_attribute_values(value)
                        if any(
                            contains_sequence(category_lemmas, normalized_surface(part, lemmatizer)[1])
                            for part in material_parts
                        ):
                            continue
                    candidates.append(Candidate(entity_type, value, value, confidence, "attribute_catalog_exact", unit))

                entities, tokens, tags, mask, overlap_rejections = resolve_entities(
                    query_normalized, candidates, lemmatizer
                )
                annotation_id = hashlib.sha256(query_preprocessed.encode("utf-8")).hexdigest()[:24]
                split = "validation"
                if category_confidence >= 0.95 and click_count >= 5:
                    grade = "A"
                elif category_confidence >= 0.90 and click_count >= 5:
                    grade = "B"
                else:
                    grade = "C"
                types = sorted({str(entity["type"]) for entity in entities})
                writer.writerow([
                    annotation_id, query_original, query_normalized, query_preprocessed,
                    category_id, category, category_path, category_confidence, click_count,
                    unique_sku_count, grade,
                    json.dumps(entities, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(types, ensure_ascii=False, separators=(",", ":")),
                    len(entities),
                    json.dumps(tokens, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(tags, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(mask, ensure_ascii=False, separators=(",", ":")),
                    "false", "true", "click_consensus+catalog_exact", split, shard,
                    boolean(has_digits_value), boolean(has_latin_value), boolean(has_cyrillic_value), boolean(mixed_value),
                ])
                counters["rows"] += 1
                counters[f"grade_{grade}"] += 1
                counters[f"split_{split}"] += 1
                counters["overlap_rejections"] += overlap_rejections
                category_counts[str(category)] += 1
                if entities:
                    counters["rows_with_entities"] += 1
                else:
                    counters["rows_without_entities"] += 1
                counters["entity_total"] += len(entities)
                for entity in entities:
                    entity_type_counts[str(entity["type"])] += 1

        connection.execute(
            f"COPY (SELECT annotation_id, query_original, query_normalized, query_preprocessed, "
            "category_id, category, category_path, TRY_CAST(category_confidence AS DOUBLE) AS category_confidence, "
            "TRY_CAST(click_count AS BIGINT) AS click_count, TRY_CAST(unique_sku_count AS BIGINT) AS unique_sku_count, "
            "quality_grade, entities_json, entity_types_json, TRY_CAST(entity_count AS INTEGER) AS entity_count, "
            "tokens_json, bio_tags_weak_json, bio_supervision_mask_json, "
            "TRY_CAST(entity_labels_complete AS BOOLEAN) AS entity_labels_complete, "
            "TRY_CAST(category_label_complete AS BOOLEAN) AS category_label_complete, label_source, split, "
            "TRY_CAST(shard AS INTEGER) AS shard, TRY_CAST(has_digits AS BOOLEAN) AS has_digits, "
            "TRY_CAST(has_latin AS BOOLEAN) AS has_latin, TRY_CAST(has_cyrillic AS BOOLEAN) AS has_cyrillic, "
            "TRY_CAST(is_mixed_language AS BOOLEAN) AS is_mixed_language "
            f"FROM read_csv({sql_string(temp_path)}, delim='\\t', header=true, all_varchar=true, quote='\"', "
            "nullstr='__SILVER_NULL_SENTINEL__')) "
            f"TO {sql_string(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)"
        )
        temp_path.unlink(missing_ok=True)

    combined = output / "silver_labels.parquet"
    parts_glob = str(shards_dir / "part-*.parquet").replace("\\", "/")
    connection.execute(
        f"COPY (SELECT * FROM read_parquet({sql_string(parts_glob)}) ORDER BY annotation_id) "
        f"TO {sql_string(combined)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    review = output / "manual_review_sample.csv"
    review_types = ["brand", "model", "os_version", *sorted(set(ATTRIBUTE_MAP.values()))]
    review_queries = [
        f"SELECT *, 1 AS review_priority FROM read_parquet({sql_string(combined)}) "
        f"ORDER BY MD5(annotation_id || '|review-general|{args.seed}') LIMIT 800"
    ]
    for entity_type in review_types:
        marker = json.dumps(entity_type, ensure_ascii=False)
        review_queries.append(
            f"SELECT *, 0 AS review_priority FROM read_parquet({sql_string(combined)}) "
            f"WHERE CONTAINS(entity_types_json, {sql_string(marker)}) "
            f"ORDER BY MD5(annotation_id || '|review-{entity_type}|{args.seed}') LIMIT 30"
        )
    review_union = " UNION ALL ".join(f"({query})" for query in review_queries)
    connection.execute(
        f"COPY (WITH candidates AS ({review_union}), deduplicated AS ("
        "SELECT *, ROW_NUMBER() OVER (PARTITION BY annotation_id ORDER BY review_priority) AS duplicate_rank "
        "FROM candidates) SELECT annotation_id, query_original, query_normalized, category, category_confidence, "
        "entities_json, quality_grade, split, '' AS reviewer_status, '' AS reviewer_comment "
        f"FROM deduplicated WHERE duplicate_rank=1 ORDER BY review_priority, MD5(annotation_id || '|review-final|{args.seed}') LIMIT 1000) "
        f"TO {sql_string(review)} (HEADER, DELIMITER ',', QUOTE '\"')"
    )

    verification = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT query_preprocessed), "
        "COUNT(*) FILTER (WHERE query_preprocessed='' OR query_preprocessed IS NULL), "
        "COUNT(*) FILTER (WHERE category_id IS NULL), "
        f"COUNT(*) FILTER (WHERE split='validation'), COUNT(*) FILTER (WHERE split<>'validation') "
        f"FROM read_parquet({sql_string(combined)})"
    ).fetchone()
    shard_sum = sum(connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0] for path in shard_paths)
    invalid_spans = 0
    # Offsets were generated from the same string. Validate a deterministic sample plus every entity JSON parse.
    for query_text, entities_json in connection.execute(
        f"SELECT query_normalized, entities_json FROM read_parquet({sql_string(combined)}) WHERE entity_count > 0"
    ).fetchall():
        for entity in json.loads(entities_json):
            if query_text[int(entity["start"]):int(entity["end"])] != entity["text"]:
                invalid_spans += 1

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "percent_requested": args.percent,
        "preprocessed_corpus_size": corpus_size,
        "rows": int(verification[0]),
        "share_of_preprocessed": 100 * int(verification[0]) / corpus_size,
        "shards": args.shards,
        "runtime_seconds": time.perf_counter() - started,
        "quality_grades": {
            "A": counters["grade_A"],
            "B": counters["grade_B"],
            "C": counters["grade_C"],
        },
        "splits": {"validation": int(verification[4])},
        "rows_with_entities": counters["rows_with_entities"],
        "rows_without_entities": counters["rows_without_entities"],
        "entity_total": counters["entity_total"],
        "entity_type_counts": dict(entity_type_counts.most_common()),
        "top_categories": dict(category_counts.most_common(20)),
        "overlap_rejections": counters["overlap_rejections"],
        "unique_queries_verified": int(verification[0]) == int(verification[1]),
        "non_validation_rows": int(verification[5]),
        "invalid_spans": invalid_spans,
        "empty_queries": int(verification[2]),
        "missing_categories": int(verification[3]),
        "shard_reconciliation": shard_sum == int(verification[0]),
        "artifacts": {
            "combined": str(combined),
            "shards": [str(path) for path in shard_paths],
            "manual_review": str(review),
        },
        "known_blockers": [
            "silver labels are not independent gold labels",
            "sku_position=0 semantics remain unconfirmed",
            "entity labels are positive-only partial annotations",
        ],
    }
    (output / "annotation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(metrics, output / "ANNOTATION_REPORT.md")
    connection.close()
    shutil.rmtree(work_dir)
    print(json.dumps({"status": "ok", "rows": metrics["rows"], "entities": metrics["entity_total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
