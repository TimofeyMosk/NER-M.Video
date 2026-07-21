"""
Диагностика: почему category-purity>=0.9 у ~81% частотных запросов, а EDA-контроль обещал ~93%.

Вывод (см. labeling/output/diag_category_purity.txt):
контрольные 93% в EDA (timofey/eda/04_mvp_signals.py) посчитаны на frequent_q[:20000] —
это ПЕРВЫЕ 20000 запросов ПО АЛФАВИТУ, а не случайные. При сортировке query_text
цифры/латиница идут раньше кириллицы, поэтому подмножество почти целиком состоит из
конкретных модельных запросов ("iphone 15 pro 256") — они почти всегда чистые.
Полная популяция частотных запросов даёт ~80%, и резолв через ИМЯ КАТЕГОРИИ совпадает
с subject_id (79.5% vs 80.0%) — значит джойн категории через каталог корректен, а не он
причина расхождения.
"""
import pickle
import numpy as np, pandas as pd

with open('cu_ws/skus.pkl', 'rb') as f:
    obj = pickle.load(f)
shop = obj['yml_catalog']['shop']
id2name = {c['@id']: c['#text'] for c in shop['categories']['category']}
sku2cat = {}
for o in shop['offers']['offer']:
    c = o.get('categories')
    nm = id2name.get(c.get('categoryId')) if isinstance(c, dict) else None
    if nm is not None:
        try:
            sku2cat[int(o['@id'])] = nm
        except (TypeError, ValueError):
            pass
s_sku = pd.Series(sku2cat, dtype='string')

df = pd.read_parquet('cu_ws/query_clicks.parquet',
                     columns=['sku_id', 'toValidUTF8(query_text)', 'sku_subject_id'])
df.columns = ['sku_id', 'query_text', 'subject_id']
df['category'] = df['sku_id'].map(s_sku)

qc = df.groupby('query_text').size()
freq_q = qc[qc >= 5].index
sub = df[df['query_text'].isin(freq_q)]


def purity(series):
    vc = series.value_counts()
    return vc.iloc[0] / vc.sum() if len(vc) else np.nan


subj_pur = sub.groupby('query_text')['subject_id'].apply(purity)
cat_pur = sub.dropna(subset=['category']).groupby('query_text')['category'].apply(purity)

print("=== Диагностика category-purity vs EDA-контроль 93% ===")
print(f"частотных запросов (>=5 кликов): {len(freq_q):,}")
print(f"[вся популяция] subject_id purity>=0.9: {(subj_pur >= 0.9).mean():.1%}")
print(f"[вся популяция] category   purity>=0.9: {(cat_pur.reindex(freq_q) >= 0.9).mean():.1%}")
print("  -> subject_id и имя категории совпадают => джойн корректен, гранулярность не при чём в агрегате")

# reproduce EDA's exact biased subset: first 20000 alphabetically
sub_eda = df[df['query_text'].isin(freq_q[:20000])]
pur_eda = sub_eda.groupby('query_text')['subject_id'].apply(purity)
n_noncyr = sum(not ('Ѐ' <= q.lstrip()[:1] <= 'ӿ') for q in freq_q[:20000] if q.strip())
print(f"\n[EDA-метод] frequent_q[:20000] (первые по алфавиту): subject purity>=0.9 = {(pur_eda >= 0.9).mean():.1%}")
print(f"  из них начинаются НЕ с кириллицы: {n_noncyr}/20000  => модельные запросы, отсюда завышенные 93%")

# the gap: queries pure by subject but not by category (sibling-category splits)
both = pd.DataFrame({'subj': subj_pur, 'cat': cat_pur.reindex(subj_pur.index)}).dropna()
split = both[(both['subj'] >= 0.9) & (both['cat'] < 0.9)]
print(f"\nзапросы subj>=0.9 но cat<0.9 (сиблинг-сплиты): {len(split):,} ({len(split)/len(both):.1%} частотных)")
