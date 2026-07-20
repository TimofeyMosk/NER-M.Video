import pandas as pd, numpy as np, json, pickle

pd.set_option('display.max_colwidth', 120)

df = pd.read_parquet('cu_ws/query_clicks.parquet')
df.columns = ['sku_id','sku_name','sku_brand_name','sku_price','sku_subject_id','sku_seo_id','query_text','sku_position']
print('shape:', df.shape)

print('\n--- nulls ---')
print(df.isnull().sum())

print('\n--- nunique ---')
for c in df.columns:
    print(f"{c}: {df[c].nunique():,}")

print('\n--- exact duplicate rows (same query+sku+position etc) ---')
dup_full = df.duplicated().sum()
print('fully duplicated rows:', dup_full, f"({dup_full/len(df):.1%})")

print('\n--- duplicate (query_text, sku_id) pairs -> click repeats ---')
pair_counts = df.groupby(['query_text','sku_id']).size()
print('unique (query,sku) pairs:', len(pair_counts))
print('pairs with >1 occurrence:', (pair_counts>1).sum(), f"({(pair_counts>1).mean():.1%})")
print(pair_counts.describe())

print('\n--- rows per query_text (clicks per query) ---')
qcounts = df['query_text'].value_counts()
print(qcounts.describe())
print('\ntop 30 queries by click rows:')
print(qcounts.head(30))

print('\n--- query length (chars/words) ---')
qtexts = df['query_text'].drop_duplicates()
wlen = qtexts.str.split().str.len()
clen = qtexts.str.len()
print('unique queries:', len(qtexts))
print('word count describe:', wlen.describe())
print('char count describe:', clen.describe())
print('\nword count value_counts (1-10):')
print(wlen.value_counts().sort_index().head(15))

print('\n--- sku_position distribution ---')
print(df['sku_position'].describe())
print(df['sku_position'].value_counts().sort_index().head(20))

print('\n--- sku_price ---')
print(df['sku_price'].describe())
print('zero price rows:', (df['sku_price']==0).sum())

print('\n--- top 30 brands by click rows ---')
print(df['sku_brand_name'].value_counts().head(30))
print('empty brand rows:', (df['sku_brand_name']=='').sum())

print('\n--- top 30 subject_id (category) by click rows ---')
print(df['sku_subject_id'].value_counts().head(30))

print('\n--- sample rows ---')
print(df.sample(10, random_state=42)[['query_text','sku_name','sku_brand_name','sku_price','sku_subject_id','sku_position']].to_string())

# save unique queries and brand list for later cross-referencing
import os
os.makedirs('eda/output', exist_ok=True)
qtexts.to_frame('query_text').to_parquet('eda/output/unique_queries.parquet')

brand_counts = df['sku_brand_name'].value_counts()
brand_counts.to_frame('count').to_parquet('eda/output/brand_counts.parquet')

# overlap with skus.pkl and sku_desc
click_skus = set(df['sku_id'].unique().tolist())
print('\nunique sku_id in clicks:', len(click_skus))

with open('cu_ws/skus.pkl','rb') as f:
    obj = pickle.load(f)
offers = obj['yml_catalog']['shop']['offers']['offer']
catalog_skus = set(int(o['@id']) for o in offers)
print('unique sku_id in catalog (skus.pkl):', len(catalog_skus))
print('overlap clicks & catalog:', len(click_skus & catalog_skus), f"({len(click_skus & catalog_skus)/len(click_skus):.1%} of click skus)")

desc_df = pd.read_parquet('cu_ws/sku_desc.parquet')
desc_skus = set(desc_df['sku_id'].unique().tolist())
print('unique sku_id in sku_desc:', len(desc_skus))
print('overlap clicks & desc:', len(click_skus & desc_skus), f"({len(click_skus & desc_skus)/len(click_skus):.1%} of click skus)")
print('overlap catalog & desc:', len(catalog_skus & desc_skus))
print('overlap all three:', len(click_skus & catalog_skus & desc_skus))

with open('eda/output/overlap_stats.json','w') as f:
    json.dump({
        'click_skus': len(click_skus),
        'catalog_skus': len(catalog_skus),
        'desc_skus': len(desc_skus),
        'click_and_catalog': len(click_skus & catalog_skus),
        'click_and_desc': len(click_skus & desc_skus),
        'catalog_and_desc': len(catalog_skus & desc_skus),
        'all_three': len(click_skus & catalog_skus & desc_skus),
    }, f, indent=1)
