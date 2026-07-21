"""
Промт 2/4: НЕЗАВИСИМЫЙ словарный источник weak-labels (gazetteer, без кликов).

Матчинг точным словарным сопоставлением (Aho-Corasick) по трём готовым справочникам:
  brand    <- eda/output/vendor_names.json      (ПОЛНЫЕ имена вендоров, без суб-токенов) + транслит
  category <- eda/output/category_names.json    (токены имён категорий, кириллица len>=4 — базис EDA)
  attr     <- eda/output/search_param_names.json (имена параметров-характеристик)
Плюс маленький ЭВРИСТИЧЕСКИЙ список транслит-пар для частотных брендов (iphone/айфон и т.п.).

Клики НЕ используются вообще — только query_text из query_clicks.parquet как источник текстов.
Все совпадения фильтруются по границам слов (без substring-мусора внутри других слов) и
схлопываются до максимальных спанов (longest-match) внутри своего типа.

Запуск:
  .venv/bin/python3 labeling/02_weak_labels_gazetteer.py 5000   # выборка
  .venv/bin/python3 labeling/02_weak_labels_gazetteer.py        # полный прогон

Артефакты:
  labeling/output/02_weak_labels_gazetteer.txt      — отчёт
  labeling/output/weak_labels_gazetteer.parquet     — датасет (только строки с >=1 совпадением)
"""
import sys, os, re, json
import numpy as np
import pandas as pd
import ahocorasick

CLICKS = 'cu_ws/query_clicks.parquet'
VENDORS = 'timofey/eda/output/vendor_names.json'
CATEGORIES = 'timofey/eda/output/category_names.json'
PARAMS = 'timofey/eda/output/search_param_names.json'
OUT_DIR = 'labeling/output'
OUT_PARQUET = os.path.join(OUT_DIR, 'weak_labels_gazetteer.parquet')

TOP10 = {
    'Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
    'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты',
}

# --- Эвристические транслит-пары для частотных брендов ---------------------
# ВНИМАНИЕ: список составлен эвристически (кириллица <- канонический латинский бренд),
# желателен ручной review. iphone/айфон — это модельная линейка Apple, а не vendor из
# справочника, добавлена намеренно, т.к. пользователи ищут её как бренд.
TRANSLIT_PAIRS = {
    'айфон': 'iphone', 'айфоны': 'iphone', 'iphone': 'iphone',
    'эпл': 'apple', 'эппл': 'apple', 'аппл': 'apple',
    'сяоми': 'xiaomi', 'ксиоми': 'xiaomi', 'ксяоми': 'xiaomi', 'шаоми': 'xiaomi',
    'самсунг': 'samsung', 'самсунк': 'samsung',
    'хуавей': 'huawei', 'хуавэй': 'huawei',
    'хонор': 'honor', 'хонер': 'honor',
    'редми': 'redmi', 'реалми': 'realme',
    'поко': 'poco', 'оппо': 'oppo', 'виво': 'vivo', 'техно': 'tecno',
    'бош': 'bosch', 'сони': 'sony', 'филипс': 'philips', 'филлипс': 'philips',
    'дайсон': 'dyson', 'хайер': 'haier', 'хаер': 'haier',
    'лджи': 'lg', 'элджи': 'lg', 'асус': 'asus', 'леново': 'lenovo', 'эйсер': 'acer',
    'сяони': 'xiaomi', 'редми ': 'redmi',
}

WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
CYR_TOKEN_RE = re.compile(r"[а-яА-ЯёЁ]+")

sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def is_wordchar(ch):
    return bool(WORD_RE.match(ch))


def load_dicts():
    vendors = json.load(open(VENDORS))          # [[name, count], ...]
    categories = json.load(open(CATEGORIES))    # [{id, parent, name}, ...]
    params = json.load(open(PARAMS))            # [[name, count], ...]
    return vendors, categories, params


def scan(query_lower, automaton):
    """Совпадения (start, end_exclusive, surface_text, type) по границам слов + longest-match."""
    hits = []
    for end_idx, (etype, key_len, canon) in automaton.iter(query_lower):
        start = end_idx - key_len + 1
        end = end_idx + 1
        # границы слова
        left_ok = start == 0 or not is_wordchar(query_lower[start - 1])
        right_ok = end == len(query_lower) or not is_wordchar(query_lower[end])
        if left_ok and right_ok:
            hits.append((start, end, query_lower[start:end], etype, canon))
    # longest-match: убрать спаны, вложенные в другой спан того же типа
    kept = []
    for h in hits:
        s, e, txt, t, c = h
        contained = any(
            (o[0] <= s and e <= o[1] and (o[1] - o[0]) > (e - s) and o[3] == t)
            for o in hits
        )
        if not contained:
            kept.append(h)
    # dedup идентичных
    seen = set()
    out = []
    for s, e, txt, t, c in sorted(kept):
        key = (s, e, t)
        if key in seen:
            continue
        seen.add(key)
        out.append({'text': txt, 'type': t, 'start': int(s), 'end': int(e), 'canon': c})
    return out


def build_automata_with_len(vendors, categories, params):
    """Как build_automata, но value = (type, key_len, canonical) — нужна длина для позиций."""
    def mk():
        return ahocorasick.Automaton()
    A_brand, A_cat, A_attr = mk(), mk(), mk()

    def add(A, key, etype, canon):
        if len(key) >= 3:
            A.add_word(key, (etype, len(key), canon))

    # BRAND: только ПОЛНЫЕ имена вендоров (фраза целиком) + транслит-пары.
    # Суб-токены многословных имён НЕ добавляем — они дают почти чистый шум
    # ('для' из 'Стиль для Дома', 'pro' из 'DORN PRO', 'смарт' из 'Смарт ТВ').
    for name, _ in vendors:
        nl = name.lower().strip()
        if len(nl) >= 3:
            add(A_brand, nl, 'BRAND', nl)
    for k, canon in TRANSLIT_PAIRS.items():
        add(A_brand, k.strip(), 'BRAND', canon)

    for c in categories:
        for tok in CYR_TOKEN_RE.findall(c['name'].lower()):
            if len(tok) >= 4:
                A_cat.add_word(tok, ('CATEGORY', len(tok), tok))

    for name, _ in params:
        nl = name.lower().strip()
        if len(nl) >= 4:
            A_attr.add_word(nl, ('ATTR', len(nl), nl))

    for A in (A_brand, A_cat, A_attr):
        A.make_automaton()
    return A_brand, A_cat, A_attr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Промт 2/4: gazetteer weak-labels (независимо от кликов) ===")
    print(f"mode: {'SAMPLE ' + str(sample_n) if sample_n else 'FULL'}")

    vendors, categories, params = load_dicts()
    print(f"[dicts] vendors={len(vendors)} categories={len(categories)} params={len(params)}")
    print(f"[translit] эвристических пар: {len(TRANSLIT_PAIRS)} (составлены вручную, желателен review)")

    A_brand, A_cat, A_attr = build_automata_with_len(vendors, categories, params)

    # только колонка query_text из кликов (клики/позиции НЕ трогаем)
    q = pd.read_parquet(CLICKS, columns=['toValidUTF8(query_text)'])
    q.columns = ['query_text']
    uniq = q['query_text'].drop_duplicates().reset_index(drop=True)
    print(f"[queries] unique query_text: {len(uniq):,}")

    # выборка EDA для сверки контрольных чисел (тот же random_state=7, 50000)
    eda_sample = uniq.sample(min(50000, len(uniq)), random_state=7)

    if sample_n:
        work = uniq.sample(min(sample_n, len(uniq)), random_state=42)
    else:
        work = uniq

    def process(series):
        rows = []
        for text in series:
            ql = text.lower()
            b = scan(ql, A_brand)
            c = scan(ql, A_cat)
            a = scan(ql, A_attr)
            rows.append((text, c, b, a))
        return rows

    # --- контрольная сверка на EDA-выборке ---
    print("\n=== coverage на EDA-выборке (50000, random_state=7) — сверка с контролем ===")
    eda_rows = process(eda_sample)
    brand_cov = np.mean([len(r[2]) > 0 for r in eda_rows])
    cat_cov = np.mean([len(r[1]) > 0 for r in eda_rows])
    attr_cov = np.mean([len(r[3]) > 0 for r in eda_rows])

    # паритет с EDA: их 57.2% посчитаны по ТОКЕНАМ vendor-имён (с шумом 'для'/'pro'/...).
    # воспроизводим ровно тот метод, чтобы показать, что база совпадает.
    vendor_tokens = set()
    for name, _ in vendors:
        for t in name.lower().split():
            if len(t) >= 3:
                vendor_tokens.add(t)
    def tok_hit(q):
        return any(t in vendor_tokens for t in WORD_RE.findall(q.lower()))
    brand_tok_cov = np.mean([tok_hit(x) for x in eda_sample])

    print(f"brand coverage (полные имена+транслит): {brand_cov:.1%}")
    print(f"  для сверки — токен-метод EDA (с шумом): {brand_tok_cov:.1%}   (EDA control ~57.2%)")
    print(f"category coverage : {cat_cov:.1%}   (EDA control ~44.9%)")
    print(f"attr coverage     : {attr_cov:.1%}   (без контроля: имена param-характеристик редки в запросах)")

    # сверка с контролем на apples-to-apples базисе: бренд — токен-метод EDA, категория — токены
    if brand_tok_cov < 0.472 or cat_cov < 0.349:  # >10pp ниже контроля
        print("\n[STOP] покрытие заметно ниже контроля — 10 непойманных примеров:")
        miss_kind = 'brand' if brand_tok_cov < 0.472 else 'category'
        idx = 2 if miss_kind == 'brand' else 1
        shown = 0
        for r in eda_rows:
            if len(r[idx]) == 0:
                print("   ", repr(r[0]))
                shown += 1
                if shown >= 10:
                    break
        return

    # --- рабочий прогон ---
    print(f"\n[run] обработка {len(work):,} запросов ...")
    rows = eda_rows if (sample_n and False) else process(work)

    df = pd.DataFrame({
        'query_text': [r[0] for r in rows],
        'category_span_matches': [r[1] for r in rows],
        'brand_span_matches': [r[2] for r in rows],
        'attr_span_matches': [r[3] for r in rows],
    })
    total = len(df)
    matched = df[(df['category_span_matches'].str.len() > 0) |
                 (df['brand_span_matches'].str.len() > 0) |
                 (df['attr_span_matches'].str.len() > 0)].copy()

    print(f"\n=== результат ===")
    print(f"обработано запросов: {total:,}")
    print(f"с >=1 совпадением любого типа: {len(matched):,} ({len(matched)/total:.1%})")
    print(f"  brand>=1   : {(df['brand_span_matches'].str.len()>0).mean():.1%}")
    print(f"  category>=1: {(df['category_span_matches'].str.len()>0).mean():.1%}")
    print(f"  attr>=1    : {(df['attr_span_matches'].str.len()>0).mean():.1%}")

    # доля запросов, чья категория-совпадение относится к топ-10 (по имени категории)
    top10_tokens = set()
    for c in categories:
        if c['name'] in TOP10:
            for tok in CYR_TOKEN_RE.findall(c['name'].lower()):
                if len(tok) >= 4:
                    top10_tokens.add(tok)
    def has_top10(spans):
        return any(s['text'] in top10_tokens for s in spans)
    print(f"  из них категория ∈ токены топ-10: "
          f"{df['category_span_matches'].apply(has_top10).mean():.1%}")

    print("\nпримеры размеченных запросов:")
    for _, row in matched.head(12).iterrows():
        b = [s['text'] for s in row['brand_span_matches']]
        c = [s['text'] for s in row['category_span_matches']]
        a = [s['text'] for s in row['attr_span_matches']]
        print(f"   {row['query_text']!r:55}  B={b} C={c} A={a}")

    print("\nтранслит-пары, добавленные эвристически (нужен ручной review):")
    for k, v in TRANSLIT_PAIRS.items():
        print(f"   {k} -> {v}")

    if not sample_n:
        matched.to_parquet(OUT_PARQUET, index=False)
        print(f"\n[saved] {OUT_PARQUET}  ({len(matched):,} rows с совпадениями из {total:,} уникальных)")
    else:
        print("\n[sample mode] parquet НЕ записан")


if __name__ == '__main__':
    main()
