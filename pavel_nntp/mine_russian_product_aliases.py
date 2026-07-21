from __future__ import annotations

import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cu_ws" / "query_clicks.parquet"
OUT_DIR = Path(__file__).resolve().parent
JSON_PATH = OUT_DIR / "russian_product_aliases.json"
CSV_PATH = OUT_DIR / "russian_product_alias_candidates.csv"
HTML_PATH = OUT_DIR / "russian_product_alias_report.html"

QUERY_COLUMN = "toValidUTF8(query_text)"
SKU_COLUMN = "sku_id"
NAME_COLUMN = "toValidUTF8(sku_name)"
BRAND_COLUMN = "toValidUTF8(sku_brand_name)"
POSITION_COLUMN = "sku_position"

DASHES_RE = re.compile(
    r"[-\u00ad\u058a\u05be\u1400\u1806\u2010-\u2015\u2212"
    r"\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\u30a0"
    r"\ufe31-\ufe32\ufe58\ufe63\uff0d]"
)
WORD_RE = re.compile(r"[а-яё]+|[a-z]+|\d+", re.IGNORECASE)
CYRILLIC_RE = re.compile(r"^[а-яё]+$", re.IGNORECASE)
LATIN_RE = re.compile(r"^[a-z]+$", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_RE = re.compile(r"(.)\1+")
VOWELS_RE = re.compile(r"[aeiouy]+")

RUS_TO_LAT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
        "й": "i", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
)

LETTER_NAMES = {
    "a": "ei", "b": "bi", "c": "si", "d": "di", "e": "i",
    "f": "ef", "g": "dzhi", "h": "eich", "i": "ai", "j": "dzhei",
    "k": "kei", "l": "el", "m": "em", "n": "en", "o": "ou",
    "p": "pi", "q": "kiu", "r": "ar", "s": "es", "t": "ti",
    "u": "yu", "v": "vi", "w": "dablyu", "x": "eks", "y": "uai",
    "z": "zed",
}

TITLE_STOP_TERMS = {
    "and", "black", "blue", "brown", "classic", "color", "digital",
    "edition", "gold", "gray", "green", "grey", "mini", "new", "orange",
    "original", "pink", "plus", "pro", "professional", "purple", "red",
    "series", "silver", "smart", "space", "true", "ultra", "white",
    "with", "wireless", "yellow", "max", "air", "good", "premium",
}

AMBIGUOUS_RUSSIAN_ALIASES = {
    "аир", "эйр", "макс", "мини", "ноут", "плюс", "про", "смарт",
    "ультра", "классик", "премиум",
}


@dataclass
class CandidateStat:
    click_rows: int = 0
    grouped_pairs: int = 0
    similarities: list[float] = field(default_factory=list)
    query_contexts: set[str] = field(default_factory=set)
    sku_ids: set[str] = field(default_factory=set)
    brands: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def add(
        self,
        count: int,
        similarity: float,
        query: str,
        sku_id: str,
        brand: str,
    ) -> None:
        self.click_rows += count
        self.grouped_pairs += 1
        self.similarities.append(similarity)
        self.query_contexts.add(query)
        self.sku_ids.add(sku_id)
        if brand:
            self.brands[brand] += count
        if query not in self.examples and len(self.examples) < 5:
            self.examples.append(query)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = DASHES_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=300_000)
def extract_russian_phrases(query: str) -> tuple[str, ...]:
    normalized = clean_text(query)
    tokens = WORD_RE.findall(normalized)
    phrases: set[str] = set()
    run: list[str] = []

    def add_run(values: list[str]) -> None:
        for start in range(len(values)):
            for size in range(1, min(3, len(values) - start) + 1):
                phrase = " ".join(values[start : start + size])
                if len(phrase.replace(" ", "")) >= 3:
                    phrases.add(phrase)

    for token in tokens:
        if CYRILLIC_RE.fullmatch(token):
            run.append(token)
        else:
            add_run(run)
            run = []
    add_run(run)
    return tuple(sorted(phrases, key=lambda item: (len(item), item)))


def canonical_latin_phrase(value: str) -> str:
    tokens = [token for token in WORD_RE.findall(clean_text(value)) if LATIN_RE.fullmatch(token)]
    return " ".join(tokens)


@lru_cache(maxsize=400_000)
def extract_english_terms(sku_name: str, brand_name: str) -> tuple[tuple[str, str], ...]:
    terms: dict[str, str] = {}
    brand = canonical_latin_phrase(brand_name)
    if len(brand.replace(" ", "")) >= 2:
        terms[brand] = "brand"
        for token in brand.split():
            if len(token) >= 2:
                terms[token] = "brand"

    title_tokens = WORD_RE.findall(clean_text(sku_name))
    latin_run: list[str] = []

    def add_run(values: list[str]) -> None:
        for token in values:
            if len(token) >= 3 and token not in TITLE_STOP_TERMS:
                terms.setdefault(token, "product_term")
        for size in (2, 3):
            for start in range(0, len(values) - size + 1):
                phrase_tokens = values[start : start + size]
                if all(token not in TITLE_STOP_TERMS for token in phrase_tokens):
                    phrase = " ".join(phrase_tokens)
                    if len(phrase.replace(" ", "")) >= 5:
                        terms.setdefault(phrase, "product_term")

    for token in title_tokens:
        if LATIN_RE.fullmatch(token):
            latin_run.append(token)
        else:
            add_run(latin_run)
            latin_run = []
    add_run(latin_run)

    return tuple(sorted(terms.items(), key=lambda item: (item[1] != "brand", item[0])))


def collapse_repeats(value: str) -> str:
    return REPEATED_RE.sub(r"\1", value)


@lru_cache(maxsize=300_000)
def russian_phonetic_variants(alias: str) -> tuple[str, ...]:
    base = alias.replace(" ", "").translate(RUS_TO_LAT)
    variants = {base, collapse_repeats(base)}
    variants.add(base.replace("ou", "o").replace("au", "a"))
    variants.add(base.replace("eia", "ia").replace("iya", "ia"))

    common_endings = ("ami", "yami", "ogo", "emu", "omu", "ami", "om", "am", "ah", "ov", "ev", "a", "u", "y", "e")
    for value in tuple(variants):
        for ending in common_endings:
            if value.endswith(ending) and len(value) - len(ending) >= 4:
                variants.add(value[: -len(ending)])
                break
    return tuple(sorted(filter(None, variants)))


def spoken_english(value: str) -> str:
    value = value.replace(" ", "")
    replacements = (
        ("tion", "shen"), ("sion", "zhen"), ("tch", "ch"),
        ("sch", "sh"), ("sh", "sh"), ("ch", "ch"), ("ph", "f"),
        ("gh", "g"), ("th", "t"), ("ck", "k"), ("qu", "kv"),
        ("ay", "ei"), ("ey", "ei"), ("oo", "u"), ("ee", "i"),
        ("w", "v"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    if value.endswith("e") and len(value) > 4:
        value = value[:-1]
    return collapse_repeats(value)


@lru_cache(maxsize=300_000)
def english_phonetic_variants(candidate: str) -> tuple[str, ...]:
    compact = candidate.replace(" ", "")
    base = spoken_english(compact)
    variants = {compact, base, collapse_repeats(compact)}

    if compact.isalpha() and 2 <= len(compact) <= 5 and compact.upper() == compact.upper():
        variants.add("".join(LETTER_NAMES[letter] for letter in compact if letter in LETTER_NAMES))

    if compact.startswith("i") and len(compact) > 4:
        variants.add("ai" + spoken_english(compact[1:]))
    if "x" in base:
        variants.add(base.replace("x", "ks"))
        variants.add(base.replace("x", "s"))
    if "y" in base:
        variants.add(base.replace("y", "i"))
        variants.add(base.replace("y", "ai"))
    if "c" in base:
        variants.add(base.replace("c", "k"))
        variants.add(base.replace("c", "s"))
    if "j" in base:
        variants.add(base.replace("j", "dzh"))
    if base.startswith("one"):
        variants.add("van" + base[3:])
    if base.startswith("a"):
        variants.add("ei" + base[1:])
    if base.endswith("me"):
        variants.add(base[:-1] + "i")

    return tuple(sorted(filter(None, variants)))


@lru_cache(maxsize=1_000_000)
def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_char in enumerate(right, 1):
        current = [row_index]
        for column_index, left_char in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def consonant_skeleton(value: str) -> str:
    return collapse_repeats(VOWELS_RE.sub("", value))


@lru_cache(maxsize=1_000_000)
def phonetic_similarity(alias: str, candidate: str) -> float:
    best = 0.0
    for russian in russian_phonetic_variants(alias):
        for english in english_phonetic_variants(candidate):
            maximum = max(len(russian), len(english), 1)
            if abs(len(russian) - len(english)) > max(3, maximum // 3):
                continue
            distance = levenshtein(russian, english)
            best = max(best, 1.0 - distance / maximum)

            russian_skeleton = consonant_skeleton(russian)
            english_skeleton = consonant_skeleton(english)
            if min(len(russian_skeleton), len(english_skeleton)) >= 2:
                skeleton_max = max(len(russian_skeleton), len(english_skeleton))
                skeleton_distance = levenshtein(russian_skeleton, english_skeleton)
                skeleton_score = 1.0 - skeleton_distance / skeleton_max
                best = max(best, skeleton_score * 0.92)
    return round(best, 4)


def match_threshold(alias: str, candidate_type: str) -> float:
    compact_length = len(alias.replace(" ", ""))
    if candidate_type == "brand":
        return 0.62 if compact_length >= 5 else 0.67
    return 0.69 if compact_length >= 5 else 0.74


def canonical_brand(value: str) -> str:
    return canonical_latin_phrase(value)


def mine_aliases() -> tuple[dict[tuple[str, str, str], CandidateStat], dict[str, int]]:
    parquet_file = pq.ParquetFile(DATA_PATH)
    stats: dict[tuple[str, str, str], CandidateStat] = defaultdict(CandidateStat)
    counters = Counter()

    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group,
            columns=[QUERY_COLUMN, SKU_COLUMN, NAME_COLUMN, BRAND_COLUMN, POSITION_COLUMN],
        )
        frame = table.to_pandas()
        frame = frame[frame[POSITION_COLUMN] > 0]
        counters["positive_rows"] += len(frame)
        if frame.empty:
            continue

        grouped = (
            frame.groupby(
                [QUERY_COLUMN, SKU_COLUMN, NAME_COLUMN, BRAND_COLUMN],
                dropna=False,
                sort=False,
            )
            .size()
            .reset_index(name="row_count")
        )
        counters["grouped_rows"] += len(grouped)

        for query, sku_id, sku_name, brand_name, row_count in grouped.itertuples(index=False, name=None):
            query_text = clean_text(query)
            phrases = extract_russian_phrases(query_text)
            if not phrases:
                continue
            terms = extract_english_terms(clean_text(sku_name), clean_text(brand_name))
            if not terms:
                continue

            brand = canonical_brand(clean_text(brand_name))
            for alias in phrases:
                best_match: tuple[float, str, str] | None = None
                for candidate, candidate_type in terms:
                    similarity = phonetic_similarity(alias, candidate)
                    if similarity < match_threshold(alias, candidate_type):
                        continue
                    adjusted = similarity + (0.015 if candidate_type == "brand" else 0.0)
                    current = (adjusted, candidate, candidate_type)
                    if best_match is None or current > best_match:
                        best_match = current

                if best_match is None:
                    continue

                adjusted, candidate, candidate_type = best_match
                similarity = adjusted - (0.015 if candidate_type == "brand" else 0.0)
                stat = stats[(alias, candidate, candidate_type)]
                stat.add(
                    count=int(row_count),
                    similarity=similarity,
                    query=query_text,
                    sku_id=str(sku_id),
                    brand=brand,
                )
                counters["matched_rows"] += int(row_count)

        if (row_group + 1) % 10 == 0 or row_group + 1 == parquet_file.num_row_groups:
            print(
                f"Обработано групп: {row_group + 1}/{parquet_file.num_row_groups}; "
                f"кандидатов: {len(stats):,}; положительных строк: {counters['positive_rows']:,}",
                flush=True,
            )

    return stats, dict(counters)


def prepare_rows(
    stats: dict[tuple[str, str, str], CandidateStat],
) -> list[dict[str, object]]:
    alias_totals: Counter[str] = Counter()
    for (alias, _, _), stat in stats.items():
        alias_totals[alias] += stat.click_rows

    rows: list[dict[str, object]] = []
    for (alias, candidate, candidate_type), stat in stats.items():
        confidence = stat.click_rows / alias_totals[alias]
        mean_similarity = sum(stat.similarities) / len(stat.similarities)
        dominant_brand = stat.brands.most_common(1)[0][0] if stat.brands else ""
        rows.append(
            {
                "alias_ru": alias,
                "canonical_en": candidate,
                "candidate_type": candidate_type,
                "click_rows": stat.click_rows,
                "query_contexts": len(stat.query_contexts),
                "unique_skus": len(stat.sku_ids),
                "confidence": round(confidence, 4),
                "mean_similarity": round(mean_similarity, 4),
                "dominant_brand": dominant_brand,
                "examples": " | ".join(stat.examples),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["alias_ru"]),
            -int(row["click_rows"]),
            -float(row["mean_similarity"]),
        )
    )

    best_for_alias: dict[str, dict[str, object]] = {}
    for row in rows:
        alias = str(row["alias_ru"])
        if alias not in best_for_alias:
            best_for_alias[alias] = row

    for row in rows:
        alias = str(row["alias_ru"])
        is_top = best_for_alias[alias] is row
        high_confidence = (
            is_top
            and int(row["click_rows"]) >= 2
            and float(row["confidence"]) >= 0.75
            and float(row["mean_similarity"]) >= 0.67
            and alias not in AMBIGUOUS_RUSSIAN_ALIASES
        )
        row["status"] = "high_confidence" if high_confidence else "needs_review"

    rows.sort(
        key=lambda row: (
            row["status"] != "high_confidence",
            -int(row["click_rows"]),
            str(row["alias_ru"]),
        )
    )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "alias_ru", "canonical_en", "candidate_type", "status", "click_rows",
        "query_contexts", "unique_skus", "confidence", "mean_similarity",
        "dominant_brand", "examples",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], counters: dict[str, int]) -> None:
    high_confidence = {
        str(row["alias_ru"]): str(row["canonical_en"])
        for row in rows
        if row["status"] == "high_confidence"
    }
    needs_review = [row for row in rows if row["status"] == "needs_review"]
    payload = {
        "metadata": {
            "source": str(DATA_PATH.relative_to(ROOT)),
            "rule": "position > 0",
            "counters": counters,
            "high_confidence_count": len(high_confidence),
            "needs_review_count": len(needs_review),
            "note": "Перед использованием словаря high_confidence рекомендуется ручная проверка.",
        },
        "high_confidence": high_confidence,
        "needs_review": needs_review,
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html(rows: list[dict[str, object]], counters: dict[str, int]) -> None:
    high_count = sum(row["status"] == "high_confidence" for row in rows)
    shown_rows = []
    for row in rows:
        status_label = "Высокая уверенность" if row["status"] == "high_confidence" else "Проверить"
        shown_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['alias_ru']))}</td>"
            f"<td>{html.escape(str(row['canonical_en']))}</td>"
            f"<td>{html.escape(str(row['candidate_type']))}</td>"
            f"<td>{status_label}</td>"
            f"<td>{int(row['click_rows']):,}</td>"
            f"<td>{int(row['query_contexts']):,}</td>"
            f"<td>{int(row['unique_skus']):,}</td>"
            f"<td>{float(row['confidence']):.1%}</td>"
            f"<td>{float(row['mean_similarity']):.1%}</td>"
            f"<td>{html.escape(str(row['examples']))}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Русскоязычные алиасы английских названий</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #20252b; font-family: Inter, system-ui, sans-serif; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #56616d; line-height: 1.5; }}
    .stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 20px 0; }}
    .stat {{ background: #fff; border: 1px solid #d9dee4; border-radius: 6px; padding: 10px 14px; }}
    .stat strong {{ display: block; font-size: 21px; }}
    .table-wrap {{ overflow: auto; max-height: 75vh; background: #fff; border: 1px solid #d9dee4; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{ padding: 9px 11px; border-bottom: 1px solid #e5e8ec; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #eef1f4; z-index: 1; font-size: 12px; }}
    td:last-child {{ white-space: normal; min-width: 360px; }}
    tr:hover td {{ background: #f8fafb; }}
  </style>
</head>
<body><main>
  <h1>Русскоязычные написания английских брендов и товаров</h1>
  <p>Кандидаты найдены во всех строках с кликом (`position &gt; 0`) через связь запроса с брендом и названием выбранного SKU. Низкоуверенные варианты сохранены для ручной проверки.</p>
  <div class="stats">
    <div class="stat"><strong>{counters.get('positive_rows', 0):,}</strong>строк с кликом</div>
    <div class="stat"><strong>{len(rows):,}</strong>соответствий</div>
    <div class="stat"><strong>{high_count:,}</strong>высокая уверенность</div>
    <div class="stat"><strong>{len(rows) - high_count:,}</strong>нужно проверить</div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Русский вариант</th><th>Канон</th><th>Тип</th><th>Статус</th><th>Строк</th><th>Контекстов</th><th>SKU</th><th>Уверенность</th><th>Сходство</th><th>Примеры запросов</th></tr></thead>
    <tbody>{''.join(shown_rows)}</tbody>
  </table></div>
</main></body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")


def main() -> None:
    stats, counters = mine_aliases()
    rows = prepare_rows(stats)
    write_csv(rows)
    write_json(rows, counters)
    write_html(rows, counters)
    print(f"Создано: {JSON_PATH}")
    print(f"Создано: {CSV_PATH}")
    print(f"Создано: {HTML_PATH}")


if __name__ == "__main__":
    main()
