import pandas as pd, numpy as np, pickle, re, json

df = pd.read_parquet('cu_ws/query_clicks.parquet')
df.columns = ['sku_id','sku_name','sku_brand_name','sku_price','sku_subject_id','sku_seo_id','query_text','sku_position']

with open('cu_ws/skus.pkl','rb') as f:
    obj = pickle.load(f)
cats = obj['yml_catalog']['shop']['categories']['category']
cat_id_to_name = {int(c['@id']): c['#text'] for c in cats}
cat_id_to_parent = {int(c['@id']): int(c['@parentId']) for c in cats if '@parentId' in c}

top_subjects = df['sku_subject_id'].value_counts().head(20)
print('--- sku_subject_id -> category name mapping check ---')
found = 0
for sid, cnt in top_subjects.items():
    name = cat_id_to_name.get(int(sid))
    if name: found += 1
    print(f"{sid}: clicks={cnt}  category_name={name}")
print(f"\nmatched {found}/{len(top_subjects)} top subject_ids directly to category ids")

all_subjects = set(df['sku_subject_id'].unique().tolist())
matched_all = sum(1 for s in all_subjects if int(s) in cat_id_to_name)
print(f"subject_id direct match to category id: {matched_all}/{len(all_subjects)} ({matched_all/len(all_subjects):.1%})")

# sku_position == 0 investigation
print('\n--- sku_position == 0 rows ---')
zero_pos = df[df['sku_position']==0]
print('count:', len(zero_pos), f"{len(zero_pos)/len(df):.1%}")
print(zero_pos[['query_text','sku_name','sku_position']].sample(5, random_state=1))
print('unique (query,sku) with pos 0:', zero_pos.duplicated(subset=['query_text','sku_id']).sum())

# query text char patterns (dedup on unique queries)
qtexts = df['query_text'].drop_duplicates()
n = len(qtexts)
has_digit = qtexts.str.contains(r'\d', regex=True).sum()
has_latin = qtexts.str.contains(r'[a-zA-Z]', regex=True).sum()
has_cyrillic = qtexts.str.contains(r'[а-яА-ЯёЁ]', regex=True).sum()
has_both_scripts = qtexts.str.contains(r'[a-zA-Z]', regex=True) & qtexts.str.contains(r'[а-яА-ЯёЁ]', regex=True)
print('\n--- query text char composition (unique queries) ---')
print(f"total unique: {n}")
print(f"contains digit: {has_digit} ({has_digit/n:.1%})")
print(f"contains latin letters: {has_latin} ({has_latin/n:.1%})")
print(f"contains cyrillic letters: {has_cyrillic} ({has_cyrillic/n:.1%})")
print(f"contains BOTH latin & cyrillic: {has_both_scripts.sum()} ({has_both_scripts.mean():.1%})")
only_latin = (qtexts.str.contains(r'[a-zA-Z]', regex=True) & ~qtexts.str.contains(r'[а-яА-ЯёЁ]', regex=True)).sum()
print(f"latin-only (no cyrillic): {only_latin} ({only_latin/n:.1%})")

# case duplication check: how many unique queries collapse when lowercased
lower_n = qtexts.str.lower().nunique()
print(f"\nunique queries: {n}, unique after lowercasing: {lower_n}  (case-only duplicates: {n - lower_n})")

# sample of digit-containing queries
print('\n--- sample queries with digits (model codes etc.) ---')
print(qtexts[qtexts.str.contains(r'\d', regex=True)].sample(15, random_state=2).tolist())

print('\n--- sample queries with mixed scripts ---')
mixed = qtexts[has_both_scripts]
print(mixed.sample(min(15,len(mixed)), random_state=3).tolist())

# brand mention rate: does query_text literally contain the clicked sku_brand_name (lowercased)?
print('\n--- does query_text contain the brand of the clicked sku (literal substring, casefolded)? ---')
sub = df[df['sku_brand_name']!=''].sample(200000, random_state=4)
contains_brand = sub.apply(lambda r: r['sku_brand_name'].lower() in r['query_text'].lower(), axis=1)
print(f"sample size={len(sub)}, brand literally in query: {contains_brand.mean():.1%}")
