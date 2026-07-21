from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cu_ws" / "query_clicks.parquet"
CATEGORIES_PATH = ROOT / "timofey" / "eda" / "output" / "category_names.json"
OUT_PATH = Path(__file__).resolve().parent / "query_clicks_analysis.html"


def fmt_int(value: float | int) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", " ")


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}".replace(".", ",")


def fmt_float(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def esc(value: object) -> str:
    return html.escape(str(value))


def table_html(df: pd.DataFrame, classes: str = "") -> str:
    return df.to_html(
        index=False,
        escape=True,
        border=0,
        classes=f"data-table {classes}".strip(),
    )


def metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f"<div class='metric-note'>{esc(note)}</div>" if note else ""
    return f"<div class='metric'><div class='metric-label'>{esc(label)}</div><div class='metric-value'>{esc(value)}</div>{note_html}</div>"


def bar_table(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    total: int | None = None,
    max_rows: int = 25,
) -> str:
    shown = df.head(max_rows).copy()
    max_value = shown[value_col].max() if len(shown) else 1
    rows = []
    for _, row in shown.iterrows():
        value = int(row[value_col])
        width = 0 if max_value == 0 else max(2, value / max_value * 100)
        share = f"<span class='muted'>{fmt_pct(value / total)}</span>" if total else ""
        rows.append(
            "<tr>"
            f"<td class='label-cell'>{esc(row[label_col])}</td>"
            f"<td class='bar-cell'><div class='bar-track'><div class='bar-fill' style='width:{width:.2f}%'></div></div></td>"
            f"<td class='num-cell'>{fmt_int(value)} {share}</td>"
            "</tr>"
        )
    return "<table class='bar-table'><tbody>" + "\n".join(rows) + "</tbody></table>"


def histogram_table(series: pd.Series, bins: list[float], labels: list[str]) -> pd.DataFrame:
    buckets = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = buckets.value_counts(sort=False)
    return pd.DataFrame(
        {
            "Диапазон": counts.index.astype(str),
            "Строк": [fmt_int(v) for v in counts.values],
            "Доля": [fmt_pct(v / len(series)) for v in counts.values],
        }
    )


def load_category_names() -> dict[int, str]:
    if not CATEGORIES_PATH.exists():
        return {}
    with CATEGORIES_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    result: dict[int, str] = {}
    for item in raw:
        try:
            result[int(item["id"])] = item["name"]
        except (KeyError, TypeError, ValueError):
            continue
    return result


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df.columns = [
        "sku_id",
        "sku_name",
        "sku_brand_name",
        "sku_price",
        "sku_subject_id",
        "sku_seo_id",
        "query_text",
        "sku_position",
    ]

    total_rows = len(df)
    total_columns = len(df.columns)
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    nulls = df.isna().sum()
    empty_strings = pd.Series(
        {
            "sku_name": (df["sku_name"] == "").sum(),
            "sku_brand_name": (df["sku_brand_name"] == "").sum(),
            "query_text": (df["query_text"].str.strip() == "").sum(),
        }
    )
    nunique = df.nunique(dropna=False)

    full_duplicates = int(df.duplicated().sum())
    pair_counts = df.groupby(["query_text", "sku_id"], sort=False).size()
    repeated_pairs = int((pair_counts > 1).sum())

    qcounts = df["query_text"].value_counts()
    top_queries = qcounts.head(30).rename_axis("Запрос").reset_index(name="Строк")
    top_queries["Доля"] = top_queries["Строк"].map(lambda x: fmt_pct(x / total_rows))

    top_queries_lower = (
        df["query_text"]
        .str.strip()
        .str.lower()
        .value_counts()
        .head(30)
        .rename_axis("Запрос normalized")
        .reset_index(name="Строк")
    )
    top_queries_lower["Доля"] = top_queries_lower["Строк"].map(lambda x: fmt_pct(x / total_rows))

    top_brands = (
        df["sku_brand_name"]
        .replace("", "<пустой бренд>")
        .value_counts()
        .head(30)
        .rename_axis("Бренд")
        .reset_index(name="Строк")
    )
    top_brands["Доля"] = top_brands["Строк"].map(lambda x: fmt_pct(x / total_rows))

    top_skus = (
        df.groupby(["sku_id", "sku_name", "sku_brand_name"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(25)
        .rename("Строк")
        .reset_index()
    )
    top_skus["sku_brand_name"] = top_skus["sku_brand_name"].replace("", "<пустой бренд>")

    category_names = load_category_names()
    top_subjects = (
        df["sku_subject_id"].value_counts().head(30).rename_axis("sku_subject_id").reset_index(name="Строк")
    )
    top_subjects["Название категории"] = top_subjects["sku_subject_id"].map(
        lambda x: category_names.get(int(x), "")
    )
    top_subjects["Доля"] = top_subjects["Строк"].map(lambda x: fmt_pct(x / total_rows))
    top_subjects = top_subjects[["sku_subject_id", "Название категории", "Строк", "Доля"]]

    price_desc = df["sku_price"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    price_stats = pd.DataFrame(
        {
            "Метрика": price_desc.index,
            "Значение": [fmt_float(v, 2) for v in price_desc.values],
        }
    )
    price_zero = int((df["sku_price"] == 0).sum())
    price_negative = int((df["sku_price"] < 0).sum())

    position_desc = df["sku_position"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    position_stats = pd.DataFrame(
        {
            "Метрика": position_desc.index,
            "Значение": [fmt_float(v, 2) for v in position_desc.values],
        }
    )
    pos_buckets = histogram_table(
        df["sku_position"],
        bins=[0, 1, 2, 3, 5, 10, 20, 50, 100, 500, np.inf],
        labels=["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"],
    )

    unique_queries = df["query_text"].drop_duplicates()
    query_stripped = unique_queries.str.strip()
    query_words = query_stripped.str.split().str.len()
    query_chars = query_stripped.str.len()
    unique_after_lower = query_stripped.str.lower().nunique()
    has_digit = query_stripped.str.contains(r"\d", regex=True, na=False)
    has_latin = query_stripped.str.contains(r"[a-zA-Z]", regex=True, na=False)
    has_cyrillic = query_stripped.str.contains(r"[а-яА-ЯёЁ]", regex=True, na=False)
    mixed_script = has_latin & has_cyrillic

    word_counts = (
        query_words.clip(upper=10)
        .replace(10, "10+")
        .value_counts()
        .sort_index(key=lambda idx: [int(x) if str(x).isdigit() else 10 for x in idx])
        .rename_axis("Слов в запросе")
        .reset_index(name="Уникальных запросов")
    )
    word_counts["Доля"] = word_counts["Уникальных запросов"].map(lambda x: fmt_pct(x / len(unique_queries)))

    query_quality = pd.DataFrame(
        [
            ["Уникальных запросов", fmt_int(len(unique_queries)), ""],
            ["Уникальных после lower/strip", fmt_int(unique_after_lower), f"-{fmt_int(len(unique_queries) - unique_after_lower)} вариантов регистра"],
            ["Содержат цифры", fmt_int(has_digit.sum()), fmt_pct(has_digit.mean())],
            ["Содержат латиницу", fmt_int(has_latin.sum()), fmt_pct(has_latin.mean())],
            ["Содержат кириллицу", fmt_int(has_cyrillic.sum()), fmt_pct(has_cyrillic.mean())],
            ["Смешивают латиницу и кириллицу", fmt_int(mixed_script.sum()), fmt_pct(mixed_script.mean())],
            ["Пустые после strip", fmt_int((query_stripped == "").sum()), fmt_pct((query_stripped == "").mean())],
            ["Средняя длина, символы", fmt_float(query_chars.mean(), 2), f"медиана {fmt_float(query_chars.median(), 2)}"],
            ["Средняя длина, слова", fmt_float(query_words.mean(), 2), f"медиана {fmt_float(query_words.median(), 2)}"],
        ],
        columns=["Метрика", "Значение", "Комментарий"],
    )

    sample_for_brand = df[df["sku_brand_name"] != ""].sample(min(300_000, (df["sku_brand_name"] != "").sum()), random_state=42)
    contains_brand = sample_for_brand.apply(
        lambda r: str(r["sku_brand_name"]).casefold() in str(r["query_text"]).casefold(),
        axis=1,
    )

    columns_table = pd.DataFrame(
        {
            "Колонка": df.columns,
            "Тип": [str(t) for t in df.dtypes],
            "Null": [fmt_int(nulls[c]) for c in df.columns],
            "Уникальных": [fmt_int(nunique[c]) for c in df.columns],
        }
    )
    columns_table["Пустых строк"] = columns_table["Колонка"].map(
        lambda c: fmt_int(empty_strings.get(c, 0)) if c in empty_strings else "-"
    )

    duplicate_table = pd.DataFrame(
        [
            ["Полные дубли строк", fmt_int(full_duplicates), fmt_pct(full_duplicates / total_rows)],
            ["Уникальные пары query_text + sku_id", fmt_int(len(pair_counts)), ""],
            ["Повторяющиеся пары query_text + sku_id", fmt_int(repeated_pairs), fmt_pct(repeated_pairs / len(pair_counts))],
            ["Максимум повторов одной пары", fmt_int(pair_counts.max()), ""],
            ["Медиана повторов пары", fmt_float(pair_counts.median(), 2), ""],
        ],
        columns=["Метрика", "Значение", "Доля"],
    )

    cards = "\n".join(
        [
            metric_card("Строк", fmt_int(total_rows), "кликовые события / строки"),
            metric_card("Колонок", fmt_int(total_columns)),
            metric_card("Уникальных запросов", fmt_int(len(unique_queries)), f"{fmt_float(total_rows / len(unique_queries), 2)} строк на запрос"),
            metric_card("Уникальных SKU", fmt_int(df["sku_id"].nunique()), ""),
            metric_card("Пустой бренд", fmt_int(empty_strings["sku_brand_name"]), fmt_pct(empty_strings["sku_brand_name"] / total_rows)),
            metric_card("Позиция 0", fmt_int((df["sku_position"] == 0).sum()), fmt_pct((df["sku_position"] == 0).mean())),
            metric_card("Цена 0", fmt_int(price_zero), fmt_pct(price_zero / total_rows)),
            metric_card("Дубликаты", fmt_int(full_duplicates), fmt_pct(full_duplicates / total_rows)),
        ]
    )

    insights = [
        f"Датасет большой: {fmt_int(total_rows)} строк, {fmt_int(df['sku_id'].nunique())} уникальных SKU и {fmt_int(len(unique_queries))} уникальных поисковых запросов.",
        f"Дубликатов много: {fmt_pct(full_duplicates / total_rows)} строк полностью повторяются. Для обучения ранжирования/NER стоит заранее решить, считать ли повтор клика весом или дедуплицировать.",
        f"Почти четверть строк имеет `sku_position = 0` ({fmt_pct((df['sku_position'] == 0).mean())}); это отдельный сильный сигнал, но его семантику лучше уточнить.",
        f"Бренд пустой в {fmt_pct(empty_strings['sku_brand_name'] / total_rows)} строк. При извлечении брендов лучше опираться не только на поле бренда в кликах.",
        f"В уникальных запросах много артикулов и моделей: цифры есть в {fmt_pct(has_digit.mean())}, латиница в {fmt_pct(has_latin.mean())}, смешанная кириллица/латиница в {fmt_pct(mixed_script.mean())}.",
        f"После нормализации регистра количество уникальных запросов уменьшается на {fmt_int(len(unique_queries) - unique_after_lower)}; простая нормализация уже заметно чистит словарь.",
        f"В выборке из {fmt_int(len(sample_for_brand))} строк с непустым брендом бренд буквально содержится в запросе примерно в {fmt_pct(contains_brand.mean())} случаев.",
    ]

    top_skus_display = top_skus.rename(
        columns={
            "sku_id": "SKU",
            "sku_name": "Название",
            "sku_brand_name": "Бренд",
        }
    )
    top_skus_display["Строк"] = top_skus_display["Строк"].map(fmt_int)

    top_subjects_display = top_subjects.copy()
    top_subjects_display["sku_subject_id"] = top_subjects_display["sku_subject_id"].map(str)
    top_subjects_display["Строк"] = top_subjects_display["Строк"].map(fmt_int)

    html_doc = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Анализ query_clicks.parquet</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #18202f;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --accent-2: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 36px 40px 22px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    main {{
      padding: 28px 40px 48px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 36px 0 14px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 24px 0 10px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    .subtitle, .muted {{
      color: var(--muted);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric-value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 700;
    }}
    .metric-note {{
      margin-top: 2px;
      color: var(--muted);
      font-size: 13px;
    }}
    section {{
      margin-top: 26px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      overflow-x: auto;
    }}
    .insights {{
      margin: 0;
      padding-left: 20px;
    }}
    .insights li {{
      margin: 8px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: #344054;
      background: #f2f5f9;
      font-weight: 650;
      white-space: nowrap;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .bar-table td {{
      border-bottom: 1px solid #e7ecf3;
    }}
    .label-cell {{
      width: 38%;
      word-break: break-word;
    }}
    .bar-cell {{
      width: 42%;
      min-width: 180px;
    }}
    .num-cell {{
      width: 20%;
      white-space: nowrap;
      text-align: right;
    }}
    .bar-track {{
      height: 12px;
      background: #eef2f6;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #22c55e);
    }}
    code {{
      background: #eef2f6;
      padding: 2px 5px;
      border-radius: 5px;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Анализ датасета query_clicks.parquet</h1>
    <div class="subtitle">Источник: <code>{esc(str(DATA_PATH.relative_to(ROOT)))}</code>. Отчет сгенерирован: {esc(generated_at)}. Примерная память датафрейма: {fmt_float(memory_mb, 1)} MB.</div>
    <div class="metrics">{cards}</div>
  </header>
  <main>
    <section class="panel">
      <h2>Ключевые выводы</h2>
      <ul class="insights">
        {''.join(f'<li>{esc(item)}</li>' for item in insights)}
      </ul>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Схема и качество данных</h2>
        {table_html(columns_table)}
      </div>
      <div class="panel">
        <h2>Дубликаты и повторы</h2>
        {table_html(duplicate_table)}
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Топ запросов</h2>
        {bar_table(top_queries, "Запрос", "Строк", total_rows, 25)}
      </div>
      <div class="panel">
        <h2>Топ запросов после lower/strip</h2>
        {bar_table(top_queries_lower, "Запрос normalized", "Строк", total_rows, 25)}
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Топ брендов</h2>
        {bar_table(top_brands, "Бренд", "Строк", total_rows, 25)}
      </div>
      <div class="panel">
        <h2>Топ категорий sku_subject_id</h2>
        {table_html(top_subjects_display)}
        <div class="note">Название подтянуто из <code>timofey/eda/output/category_names.json</code>, где есть прямое совпадение ID.</div>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Позиции SKU в выдаче</h2>
        <h3>Статистика</h3>
        {table_html(position_stats)}
        <h3>Бакеты</h3>
        {table_html(pos_buckets)}
      </div>
      <div class="panel">
        <h2>Цены SKU</h2>
        {table_html(price_stats)}
        <div class="note">Нулевых цен: {fmt_int(price_zero)} ({fmt_pct(price_zero / total_rows)}). Отрицательных цен: {fmt_int(price_negative)}.</div>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Текст запросов</h2>
        {table_html(query_quality)}
      </div>
      <div class="panel">
        <h2>Количество слов в уникальных запросах</h2>
        {table_html(word_counts)}
      </div>
    </section>

    <section class="panel">
      <h2>Топ SKU по числу строк</h2>
      {table_html(top_skus_display)}
    </section>
  </main>
</body>
</html>
"""
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
