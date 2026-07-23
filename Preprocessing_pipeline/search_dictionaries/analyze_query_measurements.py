#!/usr/bin/env python3
"""Measure how often search queries contain catalog measurement units.

The matcher uses a token trie and character-by-character tokenization. Regular
expressions are intentionally not used. The report preserves quote marks so
queries such as ``телевизор 55"`` remain observable before generic cleanup.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_dictionaries.measurement_parser import MeasurementParser, normalize_text
from search_dictionaries.units import UNIT_DEFINITIONS


DEFAULT_SOURCE = ROOT / "query_clicks.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
QUERY_COLUMN = "toValidUTF8(query_text)"


def push_example(heaps: dict[str, list[tuple[int, str]]], bucket: str, weight: int, query: str, limit: int = 12) -> None:
    item = (int(weight), str(query))
    heap = heaps[bucket]
    if item in heap:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def safe_percent(value: int | float, total: int | float) -> float:
    return 0.0 if not total else 100.0 * float(value) / float(total)


def analyze(source: Path, fetch_size: int = 10_000, limit: int | None = None) -> dict[str, object]:
    parser = MeasurementParser()
    connection = duckdb.connect()
    source_sql = "SELECT CAST(\"toValidUTF8(query_text)\" AS VARCHAR) AS query_text, COUNT(*)::BIGINT AS linked_rows FROM read_parquet(?) WHERE \"toValidUTF8(query_text)\" IS NOT NULL AND TRIM(CAST(\"toValidUTF8(query_text)\" AS VARCHAR)) <> '' GROUP BY 1"
    if limit:
        source_sql += f" LIMIT {int(limit)}"
    cursor = connection.execute(source_sql, [str(source)])

    unique = Counter()
    linked = Counter()
    mention_counts = Counter()
    unit_unique: dict[str, Counter[str]] = defaultdict(Counter)
    unit_linked: dict[str, Counter[str]] = defaultdict(Counter)
    dimension_unique: dict[str, Counter[str]] = defaultdict(Counter)
    dimension_linked: dict[str, Counter[str]] = defaultdict(Counter)
    surface_unique: Counter[tuple[str, str]] = Counter()
    surface_linked: Counter[tuple[str, str]] = Counter()
    examples: dict[str, list[tuple[int, str]]] = defaultdict(list)

    processed = 0
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        for query_raw, weight_raw in rows:
            query = str(query_raw)
            weight = int(weight_raw)
            parse_result = parser.parse(query)
            accepted = list(parse_result.candidates)
            rejected = list(parse_result.rejected)
            has_any_number = parse_result.has_any_number
            processed += 1
            unique["total"] += 1
            linked["total"] += weight
            if has_any_number:
                unique["has_any_number"] += 1
                linked["has_any_number"] += weight

            numeric_mentions = [mention for mention in accepted if mention.has_number]
            plain_mentions = [mention for mention in accepted if not mention.has_number]
            if accepted:
                unique["any_unit"] += 1
                linked["any_unit"] += weight
                push_example(examples, "any_unit", weight, query)
            if numeric_mentions:
                unique["with_number"] += 1
                linked["with_number"] += weight
                push_example(examples, "with_number", weight, query)
            if plain_mentions:
                unique["without_number"] += 1
                linked["without_number"] += weight
                push_example(examples, "without_number", weight, query)
            if numeric_mentions and plain_mentions:
                unique["both"] += 1
                linked["both"] += weight
                push_example(examples, "both", weight, query)
            elif numeric_mentions:
                unique["only_with_number"] += 1
                linked["only_with_number"] += weight
            elif plain_mentions:
                unique["only_without_number"] += 1
                linked["only_without_number"] += weight
            rejected_numeric = [mention for mention in rejected if mention.has_number]
            rejected_plain = [mention for mention in rejected if not mention.has_number]
            if rejected:
                unique["ambiguous_rejected"] += 1
                linked["ambiguous_rejected"] += weight
                if not accepted:
                    unique["only_ambiguous_rejected"] += 1
                    linked["only_ambiguous_rejected"] += weight
                    push_example(examples, "ambiguous_rejected", weight, query)
            if rejected_numeric:
                unique["ambiguous_with_number"] += 1
                linked["ambiguous_with_number"] += weight
            if rejected_plain:
                unique["ambiguous_without_number"] += 1
                linked["ambiguous_without_number"] += weight

            mention_counts["accepted"] += len(accepted)
            mention_counts["with_number"] += len(numeric_mentions)
            mention_counts["without_number"] += len(plain_mentions)
            mention_counts["ambiguous_rejected"] += len(rejected)
            mention_counts["ambiguous_with_number"] += len(rejected_numeric)
            mention_counts["ambiguous_without_number"] += len(rejected_plain)

            units_any = {mention.canonical_unit for mention in accepted}
            units_numeric = {mention.canonical_unit for mention in numeric_mentions}
            units_plain = {mention.canonical_unit for mention in plain_mentions}
            dims_any = {mention.dimension for mention in accepted}
            dims_numeric = {mention.dimension for mention in numeric_mentions}
            dims_plain = {mention.dimension for mention in plain_mentions}
            for unit_name in units_any:
                unit_unique[unit_name]["any"] += 1
                unit_linked[unit_name]["any"] += weight
            for unit_name in units_numeric:
                unit_unique[unit_name]["with_number"] += 1
                unit_linked[unit_name]["with_number"] += weight
            for unit_name in units_plain:
                unit_unique[unit_name]["without_number"] += 1
                unit_linked[unit_name]["without_number"] += weight
            for dimension in dims_any:
                dimension_unique[dimension]["any"] += 1
                dimension_linked[dimension]["any"] += weight
            for dimension in dims_numeric:
                dimension_unique[dimension]["with_number"] += 1
                dimension_linked[dimension]["with_number"] += weight
            for dimension in dims_plain:
                dimension_unique[dimension]["without_number"] += 1
                dimension_linked[dimension]["without_number"] += weight
            seen_surfaces: set[tuple[str, str]] = set()
            for mention in accepted:
                surface_key = (mention.canonical_unit, normalize_text(mention.surface).strip())
                if surface_key not in seen_surfaces:
                    surface_unique[surface_key] += 1
                    surface_linked[surface_key] += weight
                    seen_surfaces.add(surface_key)
        if processed % 100_000 < len(rows):
            print(f"processed {processed:,} unique queries", flush=True)
    connection.close()

    definitions = {str(item["canonical"]): item for item in UNIT_DEFINITIONS}
    unit_rows = []
    for canonical, counts in unit_unique.items():
        definition = definitions[canonical]
        unit_rows.append({
            "unit": canonical,
            "preferred": str(definition["preferred"]),
            "dimension": str(definition["dimension"]),
            "unique_queries_any": int(counts["any"]),
            "unique_queries_with_number": int(counts["with_number"]),
            "unique_queries_without_number": int(counts["without_number"]),
            "linked_rows_any": int(unit_linked[canonical]["any"]),
            "linked_rows_with_number": int(unit_linked[canonical]["with_number"]),
            "linked_rows_without_number": int(unit_linked[canonical]["without_number"]),
        })
    unit_rows.sort(key=lambda row: (-int(row["unique_queries_any"]), str(row["unit"])))

    dimension_rows = []
    for dimension, counts in dimension_unique.items():
        dimension_rows.append({
            "dimension": dimension,
            "unique_queries_any": int(counts["any"]),
            "unique_queries_with_number": int(counts["with_number"]),
            "unique_queries_without_number": int(counts["without_number"]),
            "linked_rows_any": int(dimension_linked[dimension]["any"]),
            "linked_rows_with_number": int(dimension_linked[dimension]["with_number"]),
            "linked_rows_without_number": int(dimension_linked[dimension]["without_number"]),
        })
    dimension_rows.sort(key=lambda row: (-int(row["unique_queries_any"]), str(row["dimension"])))

    surface_rows = [
        {
            "unit": unit_name,
            "surface": surface,
            "unique_queries": int(count),
            "linked_rows": int(surface_linked[(unit_name, surface)]),
        }
        for (unit_name, surface), count in surface_unique.most_common(100)
    ]
    example_rows = {
        bucket: [
            {"query": query, "linked_rows": weight}
            for weight, query in sorted(heap, reverse=True)
        ]
        for bucket, heap in examples.items()
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source.resolve()),
        "query_column": QUERY_COLUMN,
        "limit": limit,
        "counts": {
            "unique_queries": {key: int(value) for key, value in unique.items()},
            "linked_query_sku_rows": {key: int(value) for key, value in linked.items()},
            "mentions": {key: int(value) for key, value in mention_counts.items()},
        },
        "shares_percent": {
            "unique_queries": {
                key: safe_percent(value, unique["total"])
                for key, value in unique.items() if key != "total"
            },
            "linked_query_sku_rows": {
                key: safe_percent(value, linked["total"])
                for key, value in linked.items() if key != "total"
            },
        },
        "dictionary": {
            "unit_definitions": len(UNIT_DEFINITIONS),
            "usable_alias_patterns": parser.usable_alias_patterns,
            "token_pattern_collisions_excluded": parser.unit_pattern_collisions,
            "brand_colliding_alias_patterns_excluded": parser.brand_collisions,
            "matching": "token_trie_without_regular_expressions",
            "ambiguous_alias_policy": "single-letter alphabetic aliases are excluded from primary metrics; quote marks and 'in' require an adjacent number",
        },
        "units": unit_rows,
        "dimensions": dimension_rows,
        "surfaces": surface_rows,
        "examples": example_rows,
    }


def format_integer(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


def format_percent(value: int | float) -> str:
    return f"{float(value):.2f}%".replace(".", ",")


def svg_bars(items: list[tuple[str, float, str]], *, width: int = 860, color: str = "blue") -> str:
    if not items:
        return ""
    maximum = max(value for _, value, _ in items) or 1
    row_height = 42
    height = 24 + len(items) * row_height
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index, (label, value, display) in enumerate(items):
        y = 14 + index * row_height
        bar_width = 490 * value / maximum
        parts.append(f'<text x="0" y="{y + 16}" class="svg-label">{html.escape(label)}</text>')
        parts.append(f'<rect x="250" y="{y}" width="{bar_width:.1f}" height="24" rx="7" class="svg-bar {color}"/>')
        parts.append(f'<text x="{260 + bar_width:.1f}" y="{y + 17}" class="svg-value">{html.escape(display)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_table(headers: list[str], rows: Iterable[Iterable[object]], limit: int = 20) -> str:
    body = []
    for row in list(rows)[:limit]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render_report(metrics: dict[str, object], report_path: Path) -> None:
    unique = metrics["counts"]["unique_queries"]
    linked = metrics["counts"]["linked_query_sku_rows"]
    shares = metrics["shares_percent"]["unique_queries"]
    units = metrics["units"]
    dimensions = metrics["dimensions"]
    examples = metrics["examples"]
    upper_unique = int(unique.get("any_unit", 0)) + int(unique.get("only_ambiguous_rejected", 0))

    split_chart = svg_bars([
        ("Только число + единица", float(unique.get("only_with_number", 0)), format_integer(unique.get("only_with_number", 0))),
        ("Только единица без числа", float(unique.get("only_without_number", 0)), format_integer(unique.get("only_without_number", 0))),
        ("Оба типа в одном запросе", float(unique.get("both", 0)), format_integer(unique.get("both", 0))),
    ], color="green")
    unit_chart = svg_bars([
        (f"{row['preferred']} · {row['unit']}", float(row["unique_queries_any"]), format_integer(row["unique_queries_any"]))
        for row in units[:12]
    ])
    dimension_chart = svg_bars([
        (str(row["dimension"]), float(row["unique_queries_any"]), format_integer(row["unique_queries_any"]))
        for row in dimensions[:10]
    ], color="violet")
    unit_table = [
        (
            f"{row['preferred']} ({row['unit']})", row["dimension"],
            format_integer(row["unique_queries_any"]),
            format_integer(row["unique_queries_with_number"]),
            format_integer(row["unique_queries_without_number"]),
            format_percent(safe_percent(row["unique_queries_any"], unique["total"])),
        )
        for row in units
    ]
    surface_table = [
        (row["surface"], row["unit"], format_integer(row["unique_queries"]), format_integer(row["linked_rows"]))
        for row in metrics["surfaces"]
    ]
    numeric_examples = [
        (row["query"], format_integer(row["linked_rows"])) for row in examples.get("with_number", [])
    ]
    plain_examples = [
        (row["query"], format_integer(row["linked_rows"])) for row in examples.get("without_number", [])
    ]
    rejected_examples = [
        (row["query"], format_integer(row["linked_rows"])) for row in examples.get("ambiguous_rejected", [])
    ]

    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Единицы измерения в поисковых запросах</title>
<style>
:root{{--bg:#061018;--panel:#0b1b26;--panel2:#102735;--text:#eff9fb;--muted:#9db5bc;--cyan:#55e0d2;--blue:#5ca8ff;--green:#5de19a;--violet:#b590ff;--amber:#ffc96b;--red:#ff8196;--line:#24414b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 82% 2%,#16405a 0,transparent 31%),linear-gradient(180deg,#061018,#07131d 65%,#08121a);color:var(--text);font:15px/1.6 Inter,Segoe UI,Arial,sans-serif}}
.container{{max-width:1200px;margin:auto;padding:34px 22px 80px}}.hero{{padding:54px 0 28px}}.eyebrow{{color:var(--cyan);font-weight:750;text-transform:uppercase;letter-spacing:.13em}}
h1{{font-size:clamp(38px,6vw,72px);line-height:1.02;margin:12px 0 18px;max-width:1050px}}h2{{font-size:29px;margin:0 0 16px}}h3{{font-size:18px;margin:0 0 8px}}p{{color:var(--muted);max-width:900px}}
.lead{{font-size:18px}}section{{margin-top:42px}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}}.card{{grid-column:span 4;background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:19px;padding:22px;box-shadow:0 16px 50px #0005}}
.wide{{grid-column:span 8}}.half{{grid-column:span 6}}.full{{grid-column:1/-1}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em}}.metric{{font-size:37px;font-weight:780;line-height:1.2;margin:5px 0}}.cyan{{color:var(--cyan)}}.green-text{{color:var(--green)}}.amber{{color:var(--amber)}}
.pill{{display:inline-block;padding:4px 10px;border-radius:99px;background:#123a40;color:#8ef2e5;font-size:12px;font-weight:700;margin-right:7px}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.step{{padding:18px;border:1px solid var(--line);border-radius:15px;background:#0a1822}}
.step-no{{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--blue);font-weight:800;margin-bottom:10px}}svg{{width:100%;height:auto;overflow:visible}}.svg-label,.svg-value{{fill:var(--muted);font-size:12px}}.svg-value{{fill:var(--text);font-weight:750}}.svg-bar.blue{{fill:var(--blue)}}.svg-bar.green{{fill:var(--green)}}.svg-bar.violet{{fill:var(--violet)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}}table{{border-collapse:collapse;width:100%;min-width:760px;background:#081822}}th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#12303d;color:#d8f0f4;position:sticky;top:0}}td{{color:#bfd0d5}}
.callout{{border-left:4px solid var(--amber);padding:15px 18px;background:#211d16;border-radius:0 12px 12px 0}}.danger{{border-left-color:var(--red);background:#24171c}}code{{color:#b9f4ec;background:#081822;padding:2px 6px;border-radius:5px}}ul{{color:var(--muted)}}footer{{margin-top:50px;color:#718991}}
@media(max-width:900px){{.card,.wide,.half{{grid-column:1/-1}}.steps{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.steps{{grid-template-columns:1fr}}}}
</style></head><body><main class="container">
<header class="hero"><div class="eyebrow">M.Video · Measurement audit</div><h1>Насколько массовы единицы измерения в поисковых запросах</h1>
<p class="lead">Аудит всех <strong>{format_integer(unique['total'])}</strong> уникальных запросов. Отдельно показаны конструкции <code>число + единица</code>, упоминания единиц без числа и взвешенный срез по полному click-датасету.</p>
<span class="pill">без regex</span><span class="pill">108 типов единиц</span><span class="pill">до удаления кавычек</span></header>

<section class="grid">
<article class="card"><div class="label">Запросов с любой единицей</div><div class="metric cyan">{format_integer(unique.get('any_unit', 0))}</div><p><strong>{format_percent(shares.get('any_unit', 0))}</strong> от всех уникальных запросов.</p></article>
<article class="card"><div class="label">Число рядом с единицей</div><div class="metric green-text">{format_integer(unique.get('with_number', 0))}</div><p>{format_percent(shares.get('with_number', 0))} корпуса. Например, <code>55 дюймов</code> или <code>5000мач</code>.</p></article>
<article class="card"><div class="label">Единица без соседнего числа</div><div class="metric amber">{format_integer(unique.get('without_number', 0))}</div><p>{format_percent(shares.get('without_number', 0))}. Например, <code>кабель hdmi метры</code>.</p></article>
<article class="card"><div class="label">Полный click-датасет</div><div class="metric">{format_integer(linked['total'])}</div><p>Строки query–SKU, а не отдельные поисковые сессии.</p></article>
<article class="card"><div class="label">Связанных строк с единицами</div><div class="metric">{format_integer(linked.get('any_unit', 0))}</div><p>{format_percent(safe_percent(linked.get('any_unit', 0), linked['total']))} полного датасета.</p></article>
<article class="card"><div class="label">Неоднозначные кандидаты</div><div class="metric">{format_integer(unique.get('only_ambiguous_rejected', 0))}</div><p>Если включить их все, верхняя оценка составит {format_integer(upper_unique)} запросов ({format_percent(safe_percent(upper_unique, unique['total']))}).</p></article>
</section>

<section><h2>Как получены числа</h2><div class="steps">
<div class="step"><div class="step-no">1</div><h3>Группировка запросов</h3><p>31 млн query–SKU строк свёрнуты до уникального текста с сохранением частоты.</p></div>
<div class="step"><div class="step-no">2</div><h3>Ранняя нормализация</h3><p>NFKC, lowercase и <code>ё → е</code>. Кавычки и дефисы ещё не удаляются.</p></div>
<div class="step"><div class="step-no">3</div><h3>Token trie</h3><p>Русские и английские алиасы ищутся посимвольным токенизатором и trie без регулярных выражений.</p></div>
<div class="step"><div class="step-no">4</div><h3>Числовой контекст</h3><p>Проверяется число непосредственно слева или справа, включая слитное написание.</p></div>
</div></section>

<section class="grid"><article class="card half"><h2>Непересекающиеся группы</h2>{split_chart}</article><article class="card half"><h2>Главный вывод</h2>
<p>Число и единица — самый надёжный случай для pre-extractor. Упоминания без числа полезны как кандидаты, но требуют контекста модели.</p>
<ul><li>Только с числом: <strong>{format_integer(unique.get('only_with_number', 0))}</strong></li><li>Только без числа: <strong>{format_integer(unique.get('only_without_number', 0))}</strong></li><li>Оба типа: <strong>{format_integer(unique.get('both', 0))}</strong></li></ul></article></section>

<section class="grid"><article class="card wide"><h2>Самые частые единицы</h2>{unit_chart}</article><article class="card"><h2>По размерностям</h2>{dimension_chart}</article></section>

<section><h2>Подробно по единицам</h2><p>Числа в колонках считаются по запросам, поэтому повтор одной единицы внутри запроса не увеличивает показатель.</p>{html_table(['Единица','Размерность','Всего запросов','С числом','Без числа','Доля корпуса'], unit_table, 35)}</section>
<section><h2>Частые написания</h2>{html_table(['Поверхность','Канон','Уникальных запросов','Query–SKU строк'], surface_table, 30)}</section>

<section class="grid"><article class="card half"><h2>Примеры: число + единица</h2>{html_table(['Запрос','Query–SKU строк'], numeric_examples, 12)}</article>
<article class="card half"><h2>Примеры: единица без числа</h2>{html_table(['Запрос','Query–SKU строк'], plain_examples, 12)}</article></section>

<section><div class="callout"><strong>Консервативная неоднозначность.</strong> Однобуквенные буквенные обозначения (<code>м</code>, <code>с</code>, <code>г</code>, <code>в</code>, <code>a</code>, <code>k</code>) исключены из основного показателя даже рядом с числом: конструкции <code>A55</code>, <code>S24</code> и <code>5S</code> часто являются моделями. Алиасы единиц, совпавшие с брендами каталога — например <code>HP</code> — также исключены. Кавычки и английское <code>in</code> принимаются только рядом с числом. Отдельно отклонено {format_integer(unique.get('only_ambiguous_rejected', 0))} запросов, где был только неоднозначный кандидат.</div></section>
<section><div class="callout danger"><strong>Ограничение интерпретации.</strong> {format_integer(linked['total'])} строк полного датасета — это связи запроса с SKU/кликом, а не число пользовательских поисковых сессий. Поэтому показатель массовости по уникальным текстам является основным, а взвешенный показатель показывает представленность в click-данных.</div></section>
<section><h2>Примеры исключённой неоднозначности</h2>{html_table(['Запрос','Query–SKU строк'], rejected_examples, 12)}</section>

<section class="grid"><article class="card wide"><h2>Что делать в production</h2><ul>
<li>До модели запускать быстрый parser <code>number + unit</code> и передавать найденные spans как высокоуверенные кандидаты.</li>
<li>Не хранить все 131 тыс. комбинаций: достаточно parser чисел и lookup из 108 единиц.</li>
<li>Не удалять <code>"</code> до measurement parser — иначе теряются дюймы.</li>
<li>Упоминания без числа не использовать для безусловного model bypass.</li>
<li>Кэшировать итоговые факты с версией словаря и preprocessing.</li></ul></article>
<article class="card"><h2>Что измерять дальше</h2><ul><li>Precision/recall parser на ручном gold.</li><li>F1 отдельно по единицам.</li><li>False positive rate коротких алиасов.</li><li>Latency до и после pre-extractor.</li><li>Долю вызовов модели, которую удалось избежать.</li></ul></article></section>

<footer>Сгенерировано {html.escape(str(metrics['generated_at']))}. Источник: {html.escape(str(metrics['source']))}. Метод: token trie, без регулярных выражений.</footer>
</main></body></html>"""
    report_path.write_text(document, encoding="utf-8")


def write_unit_csv(metrics: dict[str, object], path: Path) -> None:
    fields = [
        "unit", "preferred", "dimension", "unique_queries_any",
        "unique_queries_with_number", "unique_queries_without_number",
        "linked_rows_any", "linked_rows_with_number", "linked_rows_without_number",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["units"])


def verify(metrics: dict[str, object]) -> None:
    unique = metrics["counts"]["unique_queries"]
    linked = metrics["counts"]["linked_query_sku_rows"]
    if unique["any_unit"] != unique.get("only_with_number", 0) + unique.get("only_without_number", 0) + unique.get("both", 0):
        raise AssertionError("unique query buckets do not sum to any_unit")
    if linked["any_unit"] != linked.get("only_with_number", 0) + linked.get("only_without_number", 0) + linked.get("both", 0):
        raise AssertionError("linked row buckets do not sum to any_unit")
    if unique["with_number"] != unique.get("only_with_number", 0) + unique.get("both", 0):
        raise AssertionError("numeric unit bucket mismatch")
    if unique["without_number"] != unique.get("only_without_number", 0) + unique.get("both", 0):
        raise AssertionError("non-numeric unit bucket mismatch")
    if not 0 < unique["any_unit"] <= unique["total"]:
        raise AssertionError("invalid unit coverage")


def self_test() -> None:
    parser = MeasurementParser()
    if parser.unit_pattern_collisions:
        print(f"excluded token-pattern collisions: {len(parser.unit_pattern_collisions)}")
    cases = {
        "телевизор 55 дюймов": ("inch", True),
        'телевизор 55"': ("inch", True),
        "холодильник 70см": ("cm", True),
        "аккумулятор 5000 мач": ("mah", True),
        "вес кг": ("kg", False),
        "iphone 15 s": (None, False),
        "ноутбук hp": (None, False),
        "метровый кабель": (None, False),
        "с телевизором": (None, False),
    }
    for query, (expected_unit, expected_numeric) in cases.items():
        result = parser.parse(query)
        accepted = list(result.candidates)
        if expected_unit is None:
            if accepted:
                raise AssertionError(f"unexpected mention for {query!r}: {accepted}")
            continue
        candidates = [mention for mention in accepted if mention.canonical_unit == expected_unit]
        if not candidates or candidates[0].has_number != expected_numeric:
            raise AssertionError(f"failed case {query!r}: {accepted}")
    print("SELF-TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fetch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = analyze(args.source.resolve(), args.fetch_size, args.limit)
    metrics["runtime_seconds"] = time.perf_counter() - started
    verify(metrics)
    metrics_path = output / "query_measurement_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_unit_csv(metrics, output / "query_measurement_units.csv")
    render_report(metrics, output / "query_measurement_report.html")
    print(json.dumps({
        "status": "ok",
        "metrics": str(metrics_path),
        "report": str(output / "query_measurement_report.html"),
        "unique_queries": metrics["counts"]["unique_queries"],
        "runtime_seconds": metrics["runtime_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
