"""
Промт 1/4: weak-labels для NER из лога кликов (query_clicks.parquet).

Источник слабой разметки #1 — клики. Для каждого уникального query_text определяем:
  - category_label: категория, если >=5 кликов и purity >=0.9 (доля кликов на топ-категорию),
    категория резолвится через каталог skus.pkl: sku_id -> offer.categoryId -> category #text
    (НЕ через sku_subject_id — он не совпадает с category.@id).
  - brand_label / brand_purity: топ-бренд среди кликов и его доля (БЕЗ фильтра по порогу — фильтрация позже).
  - brand_literal_match: встречается ли строка brand_label буквально в тексте запроса.
Результат ограничиваем топ-10 категориями по кликам.

Запуск:
  .venv/bin/python3 labeling/01_weak_labels_clicks.py 5000   # прогон на 5000 случайных query_text
  .venv/bin/python3 labeling/01_weak_labels_clicks.py        # полный прогон

Артефакты:
  labeling/output/01_weak_labels_clicks.txt      — этот отчёт (через редирект stdout)
  labeling/output/weak_labels_clicks.parquet     — датасет weak-labels (только полный прогон)
"""
import sys, os, pickle
import numpy as np
import pandas as pd

CLICKS = 'cu_ws/query_clicks.parquet'
SKUS = 'cu_ws/skus.pkl'
OUT_DIR = 'labeling/output'
OUT_PARQUET = os.path.join(OUT_DIR, 'weak_labels_clicks.parquet')

TOP10 = [
    'Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
    'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты',
]
MIN_CLICKS = 5
PURITY_THR = 0.9

sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = full run


def build_sku_to_category(skus_path):
    """sku_id (int) -> category #text, через offer.categories.categoryId -> category @id."""
    with open(skus_path, 'rb') as f:
        obj = pickle.load(f)
    shop = obj['yml_catalog']['shop']
    cats = shop['categories']['category']
    offers = shop['offers']['offer']
    id2name = {c['@id']: c['#text'] for c in cats}
    sku2cat = {}
    unresolved = 0
    for o in offers:
        c = o.get('categories')
        cid = c.get('categoryId') if isinstance(c, dict) else None
        name = id2name.get(cid)
        if name is None:
            unresolved += 1
            continue
        try:
            sku2cat[int(o['@id'])] = name
        except (TypeError, ValueError):
            unresolved += 1
    print(f"[catalog] categories={len(cats)} offers={len(offers)} "
          f"resolved_skus={len(sku2cat)} unresolved={unresolved}")
    # verify target names exist
    all_names = set(id2name.values())
    missing = [c for c in TOP10 if c not in all_names]
    if missing:
        print(f"[WARN] target categories not found in tree: {missing}")
    return sku2cat


def top_share_agg(df, group_col, value_col):
    """Для каждого group_col: (top value, top count, total count) по непустому value_col."""
    sub = df[[group_col, value_col]].dropna(subset=[value_col])
    sub = sub[sub[value_col] != '']
    counts = sub.groupby([group_col, value_col], observed=True).size().rename('c').reset_index()
    total = counts.groupby(group_col, observed=True)['c'].sum().rename('total')
    idx = counts.groupby(group_col, observed=True)['c'].idxmax()
    top = counts.loc[idx].set_index(group_col)
    out = top.join(total)
    out = out.rename(columns={value_col: 'label', 'c': 'top_count'})
    out['purity'] = out['top_count'] / out['total']
    return out[['label', 'purity', 'total', 'top_count']]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Промт 1/4: weak-labels из кликов ===")
    print(f"mode: {'SAMPLE ' + str(sample_n) if sample_n else 'FULL'}")

    sku2cat = build_sku_to_category(SKUS)
    s_sku = pd.Series(sku2cat, dtype='string')  # index=int sku_id, value=name

    print("\n[clicks] loading columns sku_id, brand, query_text ...")
    df = pd.read_parquet(
        CLICKS,
        columns=['sku_id', 'toValidUTF8(sku_brand_name)', 'toValidUTF8(query_text)'],
    )
    df.columns = ['sku_id', 'brand', 'query_text']
    print(f"[clicks] rows={len(df):,} unique_queries={df['query_text'].nunique():,}")

    if sample_n:
        uq = df['query_text'].drop_duplicates()
        rng = np.random.default_rng(42)
        pick = pd.Index(rng.choice(uq.to_numpy(), size=min(sample_n, len(uq)), replace=False))
        df = df[df['query_text'].isin(pick)].copy()
        print(f"[sample] {len(pick):,} queries -> {len(df):,} click rows")

    # resolve category per click row
    df['category'] = df['sku_id'].map(s_sku)
    resolved_share = df['category'].notna().mean()
    print(f"[resolve] click rows with resolved category: {resolved_share:.1%}")

    # per-query aggregates
    n_clicks = df.groupby('query_text').size().rename('n_clicks')

    cat_agg = top_share_agg(df, 'query_text', 'category')
    cat_agg = cat_agg.rename(columns={'label': 'category_label', 'purity': 'category_purity'})

    brand_agg = top_share_agg(df, 'query_text', 'brand')
    brand_agg = brand_agg.rename(columns={'label': 'brand_label', 'purity': 'brand_purity'})

    res = pd.DataFrame(n_clicks).join(cat_agg[['category_label', 'category_purity']])
    res = res.join(brand_agg[['brand_label', 'brand_purity']])

    # --- self-check numbers BEFORE thresholding/top10 filter ---
    freq = res[res['n_clicks'] >= MIN_CLICKS]
    cat_ok = freq['category_purity'].notna()
    pct_pure = (freq.loc[cat_ok, 'category_purity'] >= PURITY_THR).mean()
    print("\n=== self-check (control numbers) ===")
    print(f"queries with >={MIN_CLICKS} clicks: {len(freq):,}")
    print(f"  of those with a resolved category, share purity>=0.9: {pct_pure:.1%}  (EDA control ~93%)")
    bp = res['brand_purity'].dropna()
    print(f"brand_purity: median={bp.median():.3f} mean={bp.mean():.3f}  (EDA control median .857 mean .825)")

    # per-click literal brand match (control ~27%)
    click_lit = (
        df.dropna(subset=['brand'])
          .assign(m=lambda d: [b.lower() in q.lower() if isinstance(b, str) and b else False
                               for b, q in zip(d['brand'], d['query_text'])])
    )
    click_lit = click_lit[click_lit['brand'] != '']
    print(f"per-click literal brand match: {click_lit['m'].mean():.1%}  (EDA control ~27%)")

    if not sample_n:
        print(
            "\nNOTE по category-purity: у нас ~81% (полная популяция частотных запросов) vs\n"
            "EDA-контроль 93%. Разбор в labeling/diag_category_purity.py: контрольные 93%\n"
            "посчитаны на frequent_q[:20000] = первых 20000 запросах ПО АЛФАВИТУ (перекос в\n"
            "латиница/цифры -> модельные запросы, почти всегда чистые). На полной популяции\n"
            "subject_id и имя категории дают 79.5% vs 80.0% => джойн категории КОРРЕКТЕН,\n"
            "расхождение — из-за нерепрезентативности контрольной выборки, а не джойна.")

    # --- assign category_label only where thresholds pass ---
    keep_cat = (res['n_clicks'] >= MIN_CLICKS) & (res['category_purity'] >= PURITY_THR)
    res['category_label'] = res['category_label'].where(keep_cat)
    res['category_purity'] = res['category_purity'].where(keep_cat)

    # restrict result to top-10 categories (rows must have a top-10 category_label)
    res = res.reset_index()
    labeled = res[res['category_label'].isin(TOP10)].copy()

    # brand_literal_match per query (does brand_label appear in query_text)
    def lit(row):
        b = row['brand_label']
        if not isinstance(b, str) or not b:
            return False
        return b.lower() in row['query_text'].lower()
    labeled['brand_literal_match'] = labeled.apply(lit, axis=1)

    cols = ['query_text', 'category_label', 'category_purity', 'n_clicks',
            'brand_label', 'brand_purity', 'brand_literal_match']
    labeled = labeled[cols].sort_values('n_clicks', ascending=False).reset_index(drop=True)

    # --- report ---
    print("\n=== result: queries with a top-10 category_label ===")
    print(f"total labeled queries: {len(labeled):,}")
    print("\nper-category breakdown (count, median purity, brand_literal_match share):")
    g = labeled.groupby('category_label')
    breakdown = pd.DataFrame({
        'n_queries': g.size(),
        'median_cat_purity': g['category_purity'].median().round(3),
        'median_brand_purity': g['brand_purity'].median().round(3),
        'brand_lit_match_share': g['brand_literal_match'].mean().round(3),
        'total_clicks': g['n_clicks'].sum(),
    }).reindex(TOP10)
    print(breakdown.to_string())

    print("\nsample of labeled rows:")
    print(labeled.head(15).to_string())

    if not sample_n:
        labeled.to_parquet(OUT_PARQUET, index=False)
        print(f"\n[saved] {OUT_PARQUET}  ({len(labeled):,} rows)")
    else:
        print("\n[sample mode] parquet NOT written (run without arg for full output)")


if __name__ == '__main__':
    main()
