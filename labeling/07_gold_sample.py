"""
Формирование выборки для GOLD-сета (эталонная ручная разметка) + пред-разметка чернового BIO.

Gold нужен для ЧЕСТНОЙ приёмки: val размечен теми же weak-правилами, поэтому меряет согласие с
weak-разметкой, а не истину. Этот скрипт НЕ зависит от модели — его можно (и нужно) запускать до
обучения. На выходе — выборка ~N запросов с ЧЕРНОВЫМИ тегами (из наших словарных источников),
которые человек проверяет и правит вручную -> gold.jsonl.

ВАЖНО (bias): черновик заполнен НАШИМИ weak-правилами. Не принимать его слепо — цель ревью найти
ошибки словаря/кликов, а не подтвердить их. Часть выборки специально взята из «промахов словаря».

Состав выборки (стратификация):
  - random_head   : случайные запросы, взвешенные по частоте кликов (что реально пишут);
  - gaz_miss      : запросы с категорией, но БЕЗ единого словарного совпадения (территория модели);
  - hard_cases    : из golden_candidates.jsonl (conflict / low / редкие категории);
  - cat_balance   : добор по категориям, чтобы каждая из топ-10 была представлена.

Вход:  labeling/output/{weak_labels_clicks,weak_labels_gazetteer,weak_labels_attr}.parquet,
        labeling/output/golden_candidates.jsonl
Выход: labeling/output/gold_review.jsonl  — {query_text, tokens, tags(черновик), meta}
        labeling/output/gold_review.html   — читаемая таблица для глазной проверки

Запуск: .venv/bin/python3 labeling/07_gold_sample.py [N]   (N — целевой размер, по умолчанию 400)
"""
import os, sys, re, json, html, random
import numpy as np
import pandas as pd

OUT_DIR = 'labeling/output'
F_CLICKS = os.path.join(OUT_DIR, 'weak_labels_clicks.parquet')
F_GAZ = os.path.join(OUT_DIR, 'weak_labels_gazetteer.parquet')
F_ATTR = os.path.join(OUT_DIR, 'weak_labels_attr.parquet')
F_GOLDEN = os.path.join(OUT_DIR, 'golden_candidates.jsonl')
OUT_JSONL = os.path.join(OUT_DIR, 'gold_review.jsonl')
OUT_HTML = os.path.join(OUT_DIR, 'gold_review.html')

TOP10 = ['Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
         'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты']
# натуральные поверхности категорий для чернового выравнивания (то же, что в сборке датасета)
SURFACE = {
    'Смартфоны': ['смартфон', 'смартфоны', 'телефон', 'телефоны'],
    'Стиральные машины': ['стиральная машина', 'стиральные машины', 'стиральная машинка', 'стиралка'],
    'Холодильники': ['холодильник', 'холодильники'],
    'iPhone': ['iphone', 'айфон', 'айфоны'],
    'Телевизоры': ['телевизор', 'телевизоры'],
    'Ноутбуки': ['ноутбук', 'ноутбуки', 'ноут'],
    'Наушники': ['наушники', 'наушник'],
    'Пылесосы вертикальные': ['вертикальный пылесос', 'пылесос вертикальный', 'пылесос', 'пылесосы'],
    'Аэрогрили': ['аэрогриль', 'аэрогрили'],
    'Планшеты': ['планшет', 'планшеты'],
}
TOKEN_RE = re.compile(r'[0-9a-zа-яё]+')
PRIORITY = {'BRAND': 3, 'CATEGORY': 2, 'ATTR': 1}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
random.seed(7)


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text.lower())]


def find_surface(ql, surface):
    """Позиция surface (фраза) в ql по границам слов, exact. Возвращает (start,end)|None."""
    for m in re.finditer(re.escape(surface), ql):
        s, e = m.start(), m.end()
        left = s == 0 or not TOKEN_RE.match(ql[s - 1])
        right = e == len(ql) or not TOKEN_RE.match(ql[e])
        if left and right:
            return (s, e)
    return None


def to_bio(text, entities):
    qtokens = tokenize(text)
    owner = [None] * len(qtokens)
    for eid, (etype, s, e) in enumerate(entities):
        for ti, (tok, ts, te) in enumerate(qtokens):
            if ts < e and te > s:
                cur = owner[ti]
                if cur is None or PRIORITY[etype] > PRIORITY[cur[0]]:
                    owner[ti] = (etype, eid)
    tags, prev = [], None
    for ti in range(len(qtokens)):
        o = owner[ti]
        if o is None:
            tags.append('O'); prev = None
        else:
            tags.append(('I-' if prev == o else 'B-') + o[0]); prev = o
    return [t[0] for t in qtokens], tags


def prelabel(q, clicks_row, gaz_row, attr_spans):
    """Черновой BIO из наших источников: attr(поз.) + brand(gaz/клик-литерал) + category(surface)."""
    ql = q.lower()
    ents = []
    # ATTR — позиционированные типизированные спаны
    for s in (attr_spans if attr_spans is not None else []):
        ents.append(('ATTR', s['start'], s['end']))
    # BRAND — сначала словарный спан, иначе бренд из кликов, если буквально в тексте
    brand_hint = None
    if gaz_row is not None and len(gaz_row['brand_span_matches']):
        for s in gaz_row['brand_span_matches']:
            ents.append(('BRAND', s['start'], s['end']))
            brand_hint = s.get('canon') or s['text']
            break
    elif clicks_row is not None and isinstance(clicks_row.get('brand_label'), str) \
            and clicks_row.get('brand_literal_match'):
        sp = find_surface(ql, clicks_row['brand_label'].lower())
        if sp:
            ents.append(('BRAND', sp[0], sp[1]))
            brand_hint = clicks_row['brand_label']
    # CATEGORY — по surface-формам ярлыка из кликов
    cat_hint = clicks_row.get('category_label') if clicks_row is not None else None
    if isinstance(cat_hint, str):
        cands = sorted(set(SURFACE.get(cat_hint, []) + [cat_hint.lower()]),
                       key=lambda x: -len(x))
        for c in cands:
            sp = find_surface(ql, c)
            if sp:
                ents.append(('CATEGORY', sp[0], sp[1]))
                break
    tokens, tags = to_bio(q, ents)
    return tokens, tags, cat_hint, brand_hint


def main():
    clicks = pd.read_parquet(F_CLICKS)
    attr = pd.read_parquet(F_ATTR)
    gaz = pd.read_parquet(F_GAZ, columns=['query_text', 'brand_span_matches',
                                          'category_span_matches', 'attr_span_matches'])
    golden = [json.loads(l) for l in open(F_GOLDEN, encoding='utf-8')]

    clicks_by_q = {r['query_text']: r for r in clicks.to_dict('records')}
    attr_by_q = {r.query_text: r.attr_span_matches for r in attr.itertuples(index=False)}
    gaz_by_q = {r.query_text: {'brand_span_matches': r.brand_span_matches,
                               'category_span_matches': r.category_span_matches,
                               'attr_span_matches': r.attr_span_matches}
                for r in gaz.itertuples(index=False)}
    gaz_qset = set(gaz_by_q)
    print(f"[in] clicks={len(clicks):,} attr={len(attr):,} gazetteer={len(gaz):,} golden={len(golden):,}")

    # --- бакеты выборки ---
    picked = {}  # query_text -> bucket (первый выигрывает)

    def add(q, bucket):
        if q and q not in picked:
            picked[q] = bucket

    # 1) random_head — взвешенно по частоте кликов (log1p гасит перекос; numpy устойчивее pandas)
    w = np.log1p(clicks['n_clicks'].to_numpy(dtype=float))
    w = w / w.sum()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(clicks), size=min(150, len(clicks)), replace=False, p=w)
    for q in clicks['query_text'].to_numpy()[idx]:
        add(q, 'random_head')
    # 2) gaz_miss — есть категория, но НЕТ словарного совпадения
    miss = clicks[~clicks['query_text'].isin(gaz_qset)]
    for q in miss.sample(n=min(100, len(miss)), random_state=2)['query_text']:
        add(q, 'gaz_miss')
    # 3) hard_cases — из golden_candidates
    random.shuffle(golden)
    for g in golden[:100]:
        add(g['query_text'], 'hard_case')
    # 4) cat_balance — гарантируем покрытие всех топ-10
    for cat in TOP10:
        pool = clicks[clicks['category_label'] == cat]['query_text']
        for q in pool.sample(n=min(8, len(pool)), random_state=3):
            add(q, 'cat_balance')

    items = list(picked.items())
    random.shuffle(items)
    items = items[:N]
    print(f"[sample] всего {len(items)} запросов; состав:",
          dict(pd.Series([b for _, b in items]).value_counts()))

    # --- пред-разметка ---
    rows = []
    for q, bucket in items:
        cr = clicks_by_q.get(q)
        gr = gaz_by_q.get(q)
        tokens, tags, cat_hint, brand_hint = prelabel(q, cr, gr, attr_by_q.get(q))
        rows.append({
            'query_text': q,
            'tokens': tokens,
            'tags': tags,                 # ЧЕРНОВИК — правится вручную
            'meta': {
                'bucket': bucket,
                'gazetteer_hit': q in gaz_qset,
                'category_hint': cat_hint,
                'brand_hint': brand_hint,
                'reviewed': False,        # проставить true после проверки
            },
        })

    with open(OUT_JSONL, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    write_html(rows)
    print(f"\n[saved] {OUT_JSONL}  ({len(rows)} строк, черновой BIO)")
    print(f"[saved] {OUT_HTML}  (таблица для глазной проверки)")
    print("Дальше: проверить/поправить tags вручную, проставить meta.reviewed=true -> сохранить как gold.jsonl")


def write_html(rows):
    color = {'CATEGORY': '#1a7f37', 'BRAND': '#0969da', 'ATTR': '#9a6700'}
    parts = ["<meta charset='utf-8'><style>",
             "body{font:14px system-ui,sans-serif;margin:24px;max-width:1100px}",
             "table{border-collapse:collapse;width:100%}",
             "td,th{border:1px solid #ddd;padding:6px 8px;vertical-align:top;text-align:left}",
             "th{background:#f6f8fa}.q{color:#666;font-size:12px}",
             ".tok{display:inline-block;margin:1px 3px;padding:1px 4px;border-radius:4px}",
             "</style>",
             f"<h2>Gold review — {len(rows)} запросов (черновая разметка, правится вручную)</h2>",
             "<p>Цвета: <b style='color:#1a7f37'>CATEGORY</b>, <b style='color:#0969da'>BRAND</b>, "
             "<b style='color:#9a6700'>ATTR</b>. Задача — проверить границы и типы.</p>",
             "<table><tr><th>#</th><th>bucket</th><th>запрос → размеченные токены</th></tr>"]
    for i, r in enumerate(rows, 1):
        spans = []
        for tok, tag in zip(r['tokens'], r['tags']):
            if tag == 'O':
                spans.append(f"<span class='tok'>{html.escape(tok)}</span>")
            else:
                c = color[tag[2:]]
                spans.append(f"<span class='tok' style='background:{c}22;color:{c}'>"
                             f"{html.escape(tok)}<sub>{tag}</sub></span>")
        miss = '' if r['meta']['gazetteer_hit'] else " · <i>gaz-miss</i>"
        parts.append(f"<tr><td>{i}</td><td>{r['meta']['bucket']}{miss}</td>"
                     f"<td><div class='q'>{html.escape(r['query_text'])}</div>{' '.join(spans)}</td></tr>")
    parts.append("</table>")
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


if __name__ == '__main__':
    main()
