#!/usr/bin/env python3
"""Preprocess unique search queries and build a self-contained HTML audit report.

The normalization implementation deliberately does not use regular expressions.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable


import pymorphy3

if TYPE_CHECKING:
    import duckdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_dictionaries.measurement_parser import MeasurementParser, prepare_model_text
from search_dictionaries.color_normalizer import ColorNormalizer

DEFAULT_SOURCE = ROOT / "timofey" / "eda" / "output" / "unique_queries.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
BRANDS_PATH = ROOT / "pavel_nntp" / "data" / "dictionaries" / "brands.json"
NORMALIZATION_VERSION = "dictionary_consistent_v1"
NORMALIZATION_STEPS = (
    "unicode_nfkc",
    "yo_to_e",
    "lowercase",
    "hyphens_and_quotes_to_space",
    "whitespace_collapse",
    "safe_lemmatization",
)

HYPHEN_CHARS = frozenset(
    "-\u00ad\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015"
    "\u2212\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u301c\u3030\u30a0\ufe31\ufe32"
    "\ufe58\ufe63\uff0d"
)
QUOTE_CHARS = frozenset(
    "'\"`´ʹʻʼʽˈˊˋ˴՚՛՜՝՞՟׳״᳓᳔᳕᳖᳗᳘᳙᳜᳝᳞᳟᳚᳛᳠᳡᳢᳣᳤᳥᳦᳧᳨ᳩᳪᳫᳬ᳭ᳮᳯ"
    "‘’‚‛“”„‟‹›«»〝〞〟＂＇"
)
SEPARATOR_CHARS = HYPHEN_CHARS | QUOTE_CHARS
PROTECTED_PRODUCT_WORDS = frozenset({"мини", "макси", "ультра"})


def replace_yo(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


def lowercase(text: str) -> str:
    return text.lower()


def replace_hyphens_and_quotes(text: str) -> str:
    return "".join(" " if char in SEPARATOR_CHARS else char for char in text)


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def contains_latin(text: str) -> bool:
    for char in text:
        lowered = char.lower()
        if "a" <= lowered <= "z":
            return True
    return False


def contains_cyrillic(text: str) -> bool:
    for char in text:
        lowered = char.lower()
        if "а" <= lowered <= "я" or lowered == "ё":
            return True
    return False


def split_alphanumeric(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def load_protected_brands(path: Path = BRANDS_PATH) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    protected: set[str] = set()
    for item in data:
        if isinstance(item, list):
            brand = str(item[0])
        elif isinstance(item, dict):
            brand = str(item.get("canonical") or item.get("name") or "")
        else:
            brand = str(item)
        normalized = normalize_spaces(replace_hyphens_and_quotes(lowercase(replace_yo(brand))))
        tokens = split_alphanumeric(normalized)
        if len(tokens) == 1 and contains_cyrillic(tokens[0]):
            protected.add(tokens[0])
    return protected


@dataclass
class LemmaStats:
    token_total: int = 0
    token_changed: int = 0
    protected_brand: int = 0
    protected_latin: int = 0
    protected_digit: int = 0
    unknown: int = 0
    transformations: Counter[tuple[str, str]] = field(default_factory=Counter)


class SafeLemmatizer:
    def __init__(self, protected_brands: set[str] | None = None) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.protected_brands = protected_brands or set()
        self.stats = LemmaStats()

    @lru_cache(maxsize=500_000)
    def _lemma_decision(self, token: str) -> tuple[str, str]:
        if any(char.isdigit() for char in token):
            return token, "digit"
        if contains_latin(token):
            return token, "latin"
        if not contains_cyrillic(token) or not token.isalpha():
            return token, "other"
        if len(token) <= 2 or token in PROTECTED_PRODUCT_WORDS:
            return token, "short_or_product"
        if token in self.protected_brands:
            return token, "brand"
        if not self.morph.word_is_known(token):
            return token, "unknown"
        parses = self.morph.parse(token)
        # Product queries contain many abbreviations and ambiguous domain words.
        # If the dictionary itself offers the unchanged word as a normal form,
        # preserving it is safer than picking an unrelated higher-ranked parse.
        if any(replace_yo(parse.normal_form) == token for parse in parses):
            return token, "dictionary_fixed"
        best = parses[0]
        if best.tag.POS in {"PRTF", "PRTS"}:
            adjective = best.inflect({"nomn", "sing", "masc"})
            if adjective is not None:
                return replace_yo(adjective.word), "participle"
        return replace_yo(best.normal_form), "parsed"

    def lemmatize_token(self, token: str) -> str:
        self.stats.token_total += 1
        lemma, reason = self._lemma_decision(token)
        if reason == "brand":
            self.stats.protected_brand += 1
        elif reason == "latin":
            self.stats.protected_latin += 1
        elif reason == "digit":
            self.stats.protected_digit += 1
        elif reason == "unknown":
            self.stats.unknown += 1
        if lemma != token:
            self.stats.token_changed += 1
            self.stats.transformations[(token, lemma)] += 1
        return lemma

    def lemmatize_text(self, text: str) -> str:
        output: list[str] = []
        token: list[str] = []

        def flush() -> None:
            if token:
                output.append(self.lemmatize_token("".join(token)))
                token.clear()

        for char in text:
            if char.isalnum():
                token.append(char)
            else:
                flush()
                output.append(char)
        flush()
        return "".join(output)


@dataclass(frozen=True)
class Stages:
    original: str
    nfkc: str
    yo_equal_e: str
    lower: str
    separators: str
    spaces: str
    lemma: str


def preprocess(text: str, lemmatizer: SafeLemmatizer) -> Stages:
    original = str(text)
    nfkc = unicodedata.normalize("NFKC", original)
    yo_equal_e = replace_yo(nfkc)
    lower = lowercase(yo_equal_e)
    separators = replace_hyphens_and_quotes(lower)
    spaces = normalize_spaces(separators)
    # The morphological dictionary can reintroduce "ё" in a normal form
    # (for example, "елки" -> "ёлка"), so enforce the project contract again.
    lemma = normalize_spaces(replace_yo(lemmatizer.lemmatize_text(spaces)))
    return Stages(original, nfkc, yo_equal_e, lower, separators, spaces, lemma)


def normalize_for_lookup(text: str, lemmatizer: SafeLemmatizer) -> str:
    """Return the exact representation used by datasets and dictionaries."""
    return preprocess(text, lemmatizer).lemma


def normalization_contract() -> dict[str, object]:
    return {
        "version": NORMALIZATION_VERSION,
        "steps": list(NORMALIZATION_STEPS),
        "regular_expressions": False,
        "lookup_representation": "lemma",
        "case": "lowercase",
        "yo_equals_e": True,
        "hyphens_and_quotes": "space",
    }


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def detect_query_column(connection: duckdb.DuckDBPyConnection, source: Path) -> str:
    columns = [row[0] for row in connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
    ).fetchall()]
    for preferred in ("query_text", "query_original", "query_canonical"):
        if preferred in columns:
            return preferred
    if len(columns) == 1:
        return columns[0]
    raise ValueError(f"Не удалось определить колонку запроса. Доступны: {columns}")


def safe_percent(value: int | float, total: int | float) -> float:
    return 0.0 if not total else 100.0 * value / total


def svg_bars(items: list[tuple[str, float]], *, suffix: str = "", width: int = 780) -> str:
    if not items:
        return ""
    maximum = max(value for _, value in items) or 1
    row_height = 42
    height = 28 + len(items) * row_height
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index, (label, value) in enumerate(items):
        y = 18 + index * row_height
        bar_width = 500 * value / maximum
        parts.append(f'<text x="0" y="{y + 15}" class="svg-label">{html.escape(label)}</text>')
        parts.append(f'<rect x="220" y="{y}" width="{bar_width:.1f}" height="22" rx="6" class="svg-bar"/>')
        formatted = f"{value:,.2f}" if isinstance(value, float) and not value.is_integer() else f"{value:,.0f}"
        parts.append(f'<text x="{230 + bar_width:.1f}" y="{y + 16}" class="svg-value">{formatted}{suffix}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_table(headers: list[str], rows: Iterable[Iterable[object]], limit: int = 20) -> str:
    body = []
    for row in list(rows)[:limit]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render_report(metrics: dict[str, object], report_path: Path) -> None:
    counts = metrics["counts"]
    changes = metrics["changes"]
    lemma_stats = metrics["lemmatization"]
    examples = metrics["examples"]
    measurement = metrics.get("measurement_parser", {})
    colors = metrics.get("color_normalizer", {})
    unique_items = [(label, float(counts[key])) for label, key in (
        ("Исходные уникальные", "raw_unique"),
        ("После Unicode NFKC", "nfkc_unique"),
        ("После ё → е", "yo_unique"),
        ("После lowercase", "lower_unique"),
        ("После дефисов/кавычек", "separator_unique"),
        ("После пробелов", "space_unique"),
        ("После лемматизации", "lemma_unique"),
    )]
    changed_items = [
        ("Unicode NFKC", safe_percent(changes["nfkc"], counts["raw_unique"])),
        ("ё → е", safe_percent(changes["yo"], counts["raw_unique"])),
        ("lowercase", safe_percent(changes["lower"], counts["raw_unique"])),
        ("дефисы/кавычки", safe_percent(changes["separators"], counts["raw_unique"])),
        ("пробелы", safe_percent(changes["spaces"], counts["raw_unique"])),
        ("лемматизация", safe_percent(changes["lemma"], counts["raw_unique"])),
    ]
    top_lemmas = [(f"{row['from']} → {row['to']}", float(row["count"])) for row in lemma_stats["top_transformations"][:12]]
    measurement_types = [
        (name, float(count))
        for name, count in measurement.get("bio_entity_type_counts", {}).items()
    ]
    color_types = [
        (name, float(count))
        for name, count in colors.get("canonical_counts", {}).items()
    ][:15]
    collision_rows = [
        (row["final"], row["variants"], " · ".join(row["examples"]))
        for row in metrics["deduplication"]["collision_examples"]
    ]
    changed_rows = [(row["stage"], row["before"], row["after"]) for row in examples["stage_changes"]]
    generated = html.escape(str(metrics["generated_at"]))

    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Предобработка поисковых запросов — аудит</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--panel2:#13243a;--text:#eaf2ff;--muted:#9fb0c8;--accent:#55d6be;--accent2:#7aa2ff;--warn:#ffcc66;--bad:#ff7a90;--line:#263a55}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 85% 5%,#18365d 0,transparent 32%),var(--bg);color:var(--text);font:15px/1.6 Inter,Segoe UI,Arial,sans-serif}}
.container{{max-width:1180px;margin:auto;padding:36px 22px 80px}} .hero{{padding:54px 0 34px}} .eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase}}
h1{{font-size:clamp(36px,6vw,68px);line-height:1.03;margin:12px 0 18px;max-width:980px}} h2{{font-size:28px;margin:0 0 16px}} h3{{font-size:18px;margin:0 0 10px}} p{{color:var(--muted);max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}} .card{{grid-column:span 4;background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 14px 45px #0004}}
.wide{{grid-column:span 8}} .full{{grid-column:1/-1}} .metric{{font-size:34px;font-weight:750;margin:3px 0}} .label{{color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.08em}}
.good{{color:var(--accent)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}} section{{margin-top:42px}}
.steps{{counter-reset:step;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .step{{background:#0c192a;border:1px solid var(--line);border-radius:14px;padding:18px}}
.step:before{{counter-increment:step;content:counter(step);display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--accent2);font-weight:700;margin-bottom:10px}}
svg{{width:100%;height:auto;overflow:visible}} .svg-label,.svg-value{{fill:var(--muted);font-size:12px}} .svg-value{{fill:var(--text);font-weight:700}} .svg-bar{{fill:var(--accent2)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}} table{{border-collapse:collapse;width:100%;min-width:680px;background:#0b1727}} th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}} th{{position:sticky;top:0;background:#14263c;color:#cfe0fa}} td{{color:#c2d0e4}}
.callout{{border-left:4px solid var(--warn);padding:15px 18px;background:#211e18;border-radius:0 12px 12px 0}} code{{color:#bde9df;background:#0b1727;padding:2px 6px;border-radius:5px}} ul{{color:var(--muted)}}
footer{{margin-top:50px;color:#70839f}} @media(max-width:850px){{.card,.wide{{grid-column:1/-1}}.steps{{grid-template-columns:1fr}}}}
</style></head><body><main class="container">
<header class="hero"><div class="eyebrow">M.Video · Query understanding</div><h1>Предобработка поисковых запросов без регулярных выражений</h1>
<p>Пошаговый аудит дедупликации, <code>ё → е</code>, lowercase, удаления дефисов и кавычек, нормализации пробелов и безопасной русской лемматизации. Отчёт построен на реальном корпусе уникальных запросов.</p></header>

<section class="grid">
<article class="card"><div class="label">Строк обработано</div><div class="metric">{counts['raw_rows']:,}</div><p>До точной дедупликации источника.</p></article>
<article class="card"><div class="label">Финальных текстов</div><div class="metric good">{counts['lemma_unique']:,}</div><p>Уникальные запросы после всего pipeline.</p></article>
<article class="card"><div class="label">Схлопнуто вариантов</div><div class="metric">{counts['raw_unique'] - counts['lemma_unique']:,}</div><p>{safe_percent(counts['raw_unique'] - counts['lemma_unique'], counts['raw_unique']):.2f}% исходных уникальных форм.</p></article>
<article class="card"><div class="label">Изменено лемматизацией</div><div class="metric">{changes['lemma']:,}</div><p>{safe_percent(changes['lemma'], counts['raw_unique']):.2f}% запросов после технической нормализации.</p></article>
<article class="card"><div class="label">Изменено токенов</div><div class="metric">{lemma_stats['token_changed']:,}</div><p>Из {lemma_stats['token_total']:,} просмотренных буквенно-цифровых токенов.</p></article>
<article class="card"><div class="label">Время обработки</div><div class="metric">{metrics['runtime_seconds']:.1f} с</div><p>Локальный CPU, включая Parquet и HTML.</p></article>
<article class="card"><div class="label">Точных дублей источника</div><div class="metric">{counts['raw_rows'] - counts['raw_unique']:,}</div><p>Считаются по буквально исходной строке, до NFKC.</p></article>
</section>

<section><h2>Что именно сделано</h2><div class="steps">
<div class="step"><h3>Точная дедупликация</h3><p>Одинаковые исходные строки группируются до дорогой морфологии, но сохраняется их исходная частота.</p></div>
<div class="step"><h3>Unicode NFKC и ё → е</h3><p>Совместимые Unicode-формы унифицируются, затем обе формы русской буквы сводятся к <code>е</code>.</p></div>
<div class="step"><h3>Lowercase</h3><p>Регистр убирается до словарных сравнений. Исходный текст остаётся в audit-таблице.</p></div>
<div class="step"><h3>Дефисы и кавычки → пробел</h3><p>Используется явный Unicode-набор символов и посимвольный проход — без regex.</p></div>
<div class="step"><h3>Пробелы</h3><p>Любые последовательности whitespace превращаются в один обычный пробел, края обрезаются.</p></div>
<div class="step"><h3>Безопасная лемматизация</h3><p>Меняются только известные русские слова. Латиница, цифры, неизвестные токены и однословные бренды защищены.</p></div>
</div></section>

<section class="grid"><article class="card wide"><h2>Measurement parser: предварительная BIO-разметка</h2>
<p>Parser запускается на <code>query_model</code> до удаления кавычек и дефисов. В BIO попадают только однозначные положительные spans; остальные токены <code>O</code> имеют <code>supervision_mask=false</code>.</p>
{svg_bars(measurement_types)}</article>
<article class="card"><div class="label">Measurement-кандидатов</div><div class="metric">{measurement.get('candidate_mentions', 0):,}</div>
<p>BIO-сущностей: <strong>{measurement.get('bio_entities', 0):,}</strong><br>Запросов с BIO: <strong>{measurement.get('queries_with_bio', 0):,}</strong><br>Отклонено неоднозначных: <strong>{measurement.get('rejected_mentions', 0):,}</strong></p></article></section>

<section class="grid"><article class="card wide"><h2>Цвета: приведение к палитре из 30 классов</h2>
<p>Longest-match по каталожным и ручным алиасам выполняется на <code>query_model</code>. Исходные spans остаются неизменными для BIO, а замены записываются отдельно в <code>query_color_canonical</code>.</p>
{svg_bars(color_types)}</article>
<article class="card"><div class="label">Найдено цветов</div><div class="metric">{colors.get('candidate_mentions', 0):,}</div>
<p>Запросов с цветом: <strong>{colors.get('queries_with_candidates', 0):,}</strong><br>Изменён canonical-текст: <strong>{colors.get('queries_canonicalized', 0):,}</strong><br>Отклонено из-за конфликтов: <strong>{colors.get('rejected_mentions', 0):,}</strong></p></article></section>

<section class="grid"><article class="card wide"><h2>Число уникальных текстов по стадиям</h2>{svg_bars(unique_items)}</article>
<article class="card"><h2>Доля изменённых запросов</h2>{svg_bars(changed_items, suffix='%')}</article></section>

<section class="grid"><article class="card wide"><h2>Частые лемматизации</h2>{svg_bars(top_lemmas)}</article>
<article class="card"><h2>Что было защищено</h2><ul>
<li>Латиница: <strong>{lemma_stats['protected_latin']:,}</strong> токенов</li>
<li>Токены с цифрами: <strong>{lemma_stats['protected_digit']:,}</strong></li>
<li>Бренды: <strong>{lemma_stats['protected_brand']:,}</strong></li>
<li>Неизвестные слова: <strong>{lemma_stats['unknown']:,}</strong></li>
</ul><p>Защита уменьшает агрессивность, но сохраняет модели и товарные обозначения.</p></article></section>

<section><h2>Примеры работы отдельных стадий</h2>{html_table(['Стадия','До','После'], changed_rows, 30)}</section>
<section><h2>Что схлопнулось после нормализации</h2><p>Такие группы объясняют реальный эффект семантической дедупликации.</p>{html_table(['Финальный текст','Вариантов','Примеры'], collision_rows, 25)}</section>

<section class="grid"><article class="card wide"><h2>Что работает</h2><ul>
<li>Техническая нормализация детерминирована и не зависит от обучающей выборки.</li>
<li>Модели и латинские бренды не проходят через русский морфологический словарь.</li>
<li>Все стадии сохранены, поэтому любое изменение можно объяснить и откатить.</li>
<li>Финальная дедупликация сокращает повторное обучение на орфографических вариантах.</li>
</ul></article>
<article class="card"><h2>Что не решено</h2><ul>
<li>Словарная лемматизация не разрешает контекстную неоднозначность.</li>
<li>Удаление дефиса может повредить часть моделей: оригинал нужно хранить всегда.</li>
<li>Удаление <code>"</code> стирает обозначение дюймов в запросах вроде <code>телевизор 55"</code>.</li>
<li>Смешанная кириллица/латиница пока не исправляется.</li>
<li>Опечатки и транслитерация не нормализуются.</li>
</ul></article></section>

<section><div class="callout"><strong>Рекомендация.</strong> Для category classifier можно использовать лемматизированный текст как дополнительный канал признаков, но не заменять им исходно нормализованный текст. Для NER и моделей/серий основной вход должен сохранять поверхностную форму и offsets. Двойную кавычку после числа лучше в следующей версии преобразовывать в токен <code>дюйм</code>, а не удалять.</div></section>

<footer>Сгенерировано {generated}. Источник: {html.escape(str(metrics['source']))}. Лимит: {metrics['limit'] or 'полный корпус'}.</footer>
</main></body></html>"""
    report_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--column")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fetch-size", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    import duckdb

    args = parse_args()
    started = time.perf_counter()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)

    connection = duckdb.connect()
    column = args.column or detect_query_column(connection, source)
    identifier = quote_identifier(column)
    limit_sql = f" LIMIT {int(args.limit)}" if args.limit else ""
    query = (
        f"SELECT CAST({identifier} AS VARCHAR) AS query_original, COUNT(*)::BIGINT AS raw_count "
        f"FROM read_parquet(?) WHERE {identifier} IS NOT NULL "
        f"GROUP BY {identifier} ORDER BY {identifier}{limit_sql}"
    )
    cursor = connection.execute(query, [str(source)])

    lemmatizer = SafeLemmatizer(load_protected_brands())
    measurement_parser = MeasurementParser()
    color_normalizer = ColorNormalizer()
    counters = Counter()
    measurement_entity_counts: Counter[str] = Counter()
    color_canonical_counts: Counter[str] = Counter()
    stage_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    temp_tsv = output_dir / "preprocessing_rows.tmp"
    fields = [
        "query_original", "raw_count", "query_model", "nfkc", "yo_equal_e", "lower",
        "separators", "spaces", "lemma", "measurement_candidates_json",
        "measurement_bio_entities_json", "measurement_tokens_json",
        "measurement_bio_tags_json", "measurement_bio_mask_json", "measurement_rejected_count",
        "query_color_canonical", "color_candidates_json", "color_bio_entities_json",
        "color_tokens_json", "color_bio_tags_json", "color_bio_mask_json", "color_rejected_count",
    ]

    with temp_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fields)
        processed = 0
        while rows := cursor.fetchmany(args.fetch_size):
            for raw, raw_count in rows:
                stages = preprocess(raw, lemmatizer)
                model_text = prepare_model_text(stages.original)
                measurement_result = measurement_parser.parse(model_text)
                color_result = color_normalizer.parse(model_text)
                measurement_candidates = [item.to_dict() for item in measurement_result.candidates]
                measurement_entities = [item.to_dict() for item in measurement_result.candidates if item.bio_eligible]
                if measurement_entities:
                    measurement_tokens, measurement_tags, measurement_mask = measurement_parser.word_bio(
                        model_text, measurement_result.candidates
                    )
                else:
                    measurement_tokens, measurement_tags, measurement_mask = [], [], []
                color_candidates = [item.to_dict() for item in color_result.candidates]
                color_entities = [item.to_dict() for item in color_result.candidates if item.bio_eligible]
                if color_entities:
                    color_tokens, color_tags, color_mask = color_normalizer.word_bio(
                        model_text, color_result.candidates
                    )
                else:
                    color_tokens, color_tags, color_mask = [], [], []
                values = [
                    stages.original, raw_count, model_text, stages.nfkc, stages.yo_equal_e, stages.lower,
                    stages.separators, stages.spaces, stages.lemma,
                    json.dumps(measurement_candidates, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(measurement_entities, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(measurement_tokens, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(measurement_tags, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(measurement_mask, ensure_ascii=False, separators=(",", ":")),
                    len(measurement_result.rejected),
                    color_result.canonical_text,
                    json.dumps(color_candidates, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(color_entities, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(color_tokens, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(color_tags, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(color_mask, ensure_ascii=False, separators=(",", ":")),
                    len(color_result.rejected),
                ]
                writer.writerow(values)
                counters["raw_rows"] += int(raw_count)
                counters["raw_unique"] += 1
                comparisons = (
                    ("nfkc", stages.original, stages.nfkc),
                    ("yo", stages.nfkc, stages.yo_equal_e),
                    ("lower", stages.yo_equal_e, stages.lower),
                    ("separators", stages.lower, stages.separators),
                    ("spaces", stages.separators, stages.spaces),
                    ("lemma", stages.spaces, stages.lemma),
                )
                for name, before, after in comparisons:
                    if before != after:
                        counters[name] += 1
                        if len(stage_examples[name]) < 7:
                            stage_examples[name].append({"stage": name, "before": before, "after": after})
                if not stages.lemma:
                    counters["empty_final"] += 1
                counters["measurement_candidates"] += len(measurement_result.candidates)
                counters["measurement_bio_entities"] += len(measurement_entities)
                counters["measurement_rejected"] += len(measurement_result.rejected)
                if measurement_result.candidates:
                    counters["queries_with_measurement_candidates"] += 1
                if measurement_entities:
                    counters["queries_with_measurement_bio"] += 1
                for entity in measurement_entities:
                    measurement_entity_counts[str(entity["entity_type"])] += 1
                counters["color_candidates"] += len(color_result.candidates)
                counters["color_rejected"] += len(color_result.rejected)
                if color_result.candidates:
                    counters["queries_with_colors"] += 1
                if color_result.canonical_text != model_text:
                    counters["queries_color_canonicalized"] += 1
                for entity in color_entities:
                    color_canonical_counts[str(entity["canonical"])] += 1
            processed += len(rows)
            if processed % 50_000 < len(rows):
                print(f"processed {processed:,} unique queries", flush=True)

    audit_path = output_dir / "preprocessed_queries_audit.parquet"
    final_path = output_dir / "preprocessed_queries.parquet"
    connection.execute(
        f"CREATE OR REPLACE VIEW preprocessing AS SELECT * FROM read_csv({quote_sql_string(temp_tsv)}, "
        "delim='\\t', header=true, all_varchar=true, quote='\"', "
        "nullstr='__QUERY_PREPROCESSING_NULL_SENTINEL__')"
    )
    connection.execute(
        f"COPY (SELECT query_original, TRY_CAST(raw_count AS BIGINT) AS raw_count, query_model, "
        "nfkc, yo_equal_e, lower, separators, spaces, lemma, measurement_candidates_json, "
        "measurement_bio_entities_json, measurement_tokens_json, measurement_bio_tags_json, "
        "measurement_bio_mask_json, TRY_CAST(measurement_rejected_count AS INTEGER) AS measurement_rejected_count, "
        "query_color_canonical, color_candidates_json, color_bio_entities_json, color_tokens_json, "
        "color_bio_tags_json, color_bio_mask_json, TRY_CAST(color_rejected_count AS INTEGER) AS color_rejected_count "
        f"FROM preprocessing) TO {quote_sql_string(audit_path)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)",
    )
    connection.execute(
        "COPY (SELECT lemma AS query_preprocessed, SUM(TRY_CAST(raw_count AS BIGINT))::BIGINT AS source_row_count, "
        "COUNT(*)::BIGINT AS original_variant_count, LIST_SLICE(LIST(query_original ORDER BY query_original), 1, 5) AS original_examples, "
        "ARG_MIN(query_model, query_original) AS query_model_example, "
        "ARG_MIN(measurement_candidates_json, query_original) AS measurement_candidates_example_json, "
        "ARG_MIN(measurement_bio_entities_json, query_original) AS measurement_bio_entities_example_json, "
        "ARG_MIN(measurement_bio_tags_json, query_original) AS measurement_bio_tags_example_json, "
        "ARG_MIN(query_color_canonical, query_original) AS query_color_canonical_example, "
        "ARG_MIN(color_bio_entities_json, query_original) AS color_bio_entities_example_json, "
        "ARG_MIN(color_bio_tags_json, query_original) AS color_bio_tags_example_json "
        f"FROM preprocessing WHERE lemma <> '' GROUP BY lemma) TO {quote_sql_string(final_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)",
    )

    distinct = connection.execute(
        "SELECT COUNT(DISTINCT query_original), COUNT(DISTINCT nfkc), COUNT(DISTINCT yo_equal_e), COUNT(DISTINCT lower), "
        "COUNT(DISTINCT separators), COUNT(DISTINCT spaces), "
        "COUNT(DISTINCT CASE WHEN lemma <> '' THEN lemma END) FROM preprocessing"
    ).fetchone()
    collision_data = connection.execute(
        "SELECT lemma, COUNT(*) AS variants, LIST_SLICE(LIST(query_original ORDER BY query_original), 1, 5) AS examples "
        "FROM preprocessing WHERE lemma <> '' GROUP BY lemma HAVING COUNT(*) > 1 ORDER BY variants DESC, lemma LIMIT 30"
    ).fetchall()
    collision_groups = connection.execute(
        "SELECT COUNT(*) FROM (SELECT lemma FROM preprocessing WHERE lemma <> '' GROUP BY lemma HAVING COUNT(*) > 1)"
    ).fetchone()[0]

    sample_path = output_dir / "preprocessing_samples.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "before", "after"])
        for stage in ("nfkc", "yo", "lower", "separators", "spaces", "lemma"):
            for row in stage_examples[stage]:
                writer.writerow([row["stage"], row["before"], row["after"]])

    top_transformations = [
        {"from": source_token, "to": lemma, "count": count}
        for (source_token, lemma), count in lemmatizer.stats.transformations.most_common(50)
    ]
    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "normalization_contract": normalization_contract(),
        "source": str(source),
        "source_column": column,
        "limit": args.limit,
        "runtime_seconds": time.perf_counter() - started,
        "counts": {
            "raw_rows": counters["raw_rows"],
            "raw_unique": int(distinct[0]),
            "nfkc_unique": int(distinct[1]),
            "yo_unique": int(distinct[2]),
            "lower_unique": int(distinct[3]),
            "separator_unique": int(distinct[4]),
            "space_unique": int(distinct[5]),
            "lemma_unique": int(distinct[6]),
            "empty_final": counters["empty_final"],
        },
        "changes": {name: counters[name] for name in ("nfkc", "yo", "lower", "separators", "spaces", "lemma")},
        "lemmatization": {
            "token_total": lemmatizer.stats.token_total,
            "token_changed": lemmatizer.stats.token_changed,
            "protected_brand": lemmatizer.stats.protected_brand,
            "protected_latin": lemmatizer.stats.protected_latin,
            "protected_digit": lemmatizer.stats.protected_digit,
            "unknown": lemmatizer.stats.unknown,
            "cache": str(lemmatizer._lemma_decision.cache_info()),
            "top_transformations": top_transformations,
        },
        "deduplication": {
            "collision_groups": int(collision_groups),
            "collapsed_unique_variants": int(distinct[0] - distinct[6]),
            "collision_examples": [
                {"final": row[0], "variants": int(row[1]), "examples": row[2]} for row in collision_data
            ],
        },
        "examples": {
            "stage_changes": [row for stage in ("nfkc", "yo", "lower", "separators", "spaces", "lemma") for row in stage_examples[stage]],
        },
        "artifacts": {
            "audit_parquet": str(audit_path),
            "final_parquet": str(final_path),
            "samples_csv": str(sample_path),
        },
        "measurement_parser": {
            **measurement_parser.metadata(),
            "candidate_mentions": counters["measurement_candidates"],
            "bio_entities": counters["measurement_bio_entities"],
            "rejected_mentions": counters["measurement_rejected"],
            "queries_with_candidates": counters["queries_with_measurement_candidates"],
            "queries_with_bio": counters["queries_with_measurement_bio"],
            "bio_entity_type_counts": dict(measurement_entity_counts.most_common()),
            "span_text": "query_model",
        },
        "color_normalizer": {
            **color_normalizer.metadata(),
            "candidate_mentions": counters["color_candidates"],
            "bio_entities": counters["color_candidates"],
            "rejected_mentions": counters["color_rejected"],
            "queries_with_candidates": counters["queries_with_colors"],
            "queries_with_bio": counters["queries_with_colors"],
            "queries_canonicalized": counters["queries_color_canonicalized"],
            "canonical_counts": dict(color_canonical_counts.most_common()),
        },
    }
    metrics_path = output_dir / "preprocessing_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(metrics, output_dir / "text_preprocessing_report.html")
    temp_tsv.unlink(missing_ok=True)
    connection.close()
    print(json.dumps({"status": "ok", "metrics": str(metrics_path), "rows": counters["raw_unique"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
