import pandas as pd, numpy as np, pickle, re, json, collections

df = pd.read_parquet('cu_ws/query_clicks.parquet')
df.columns = ['sku_id','sku_name','sku_brand_name','sku_price','sku_subject_id','sku_seo_id','query_text','sku_position']

with open('cu_ws/skus.pkl','rb') as f:
    obj = pickle.load(f)
offers = obj['yml_catalog']['shop']['offers']['offer']
cats = obj['yml_catalog']['shop']['categories']['category']
cat_names = [c['#text'] for c in cats]
vendors = sorted(set(o['vendor'].lower() for o in offers if o.get('vendor')))

# attribute value vocab size for a few key "facts"
value_sets = collections.defaultdict(set)
target_facts = ['Цвет_search','Материал_search','Материал корпуса_search','Форма_search','Страна_search']
for o in offers:
    params = o.get('param', [])
    if isinstance(params, dict): params = [params]
    for p in params:
        if p.get('@name') in target_facts:
            value_sets[p['@name']].add(p.get('#text','').strip().lower())

print('--- cardinality of value sets for closed-ish facts ---')
for name in target_facts:
    vals = value_sets[name]
    print(f"{name}: {len(vals)} unique values; sample: {list(vals)[:8]}")

# gazetteer coverage on a sample of unique queries
qtexts = df['query_text'].drop_duplicates()
sample = qtexts.sample(min(50000, len(qtexts)), random_state=7)

vendor_set = set(vendors)
def brand_hit(q):
    ql = q.lower()
    return any(v in ql for v in vendor_set if len(v) >= 3)  # cheap substring scan is slow; will optimize below

# faster: tokenize query, check token/bigram membership against vendor set (exact) instead of full substring scan over 6765 vendors x 50k queries
vendor_tokens = set()
for v in vendors:
    for tok in re.split(r'\s+', v):
        if len(tok) >= 3:
            vendor_tokens.add(tok)

def tokenize(q):
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", q.lower())

hits_brand_token = 0
hits_digit = 0
for q in sample:
    toks = tokenize(q)
    if any(t in vendor_tokens for t in toks):
        hits_brand_token += 1
    if any(t.isdigit() or any(ch.isdigit() for ch in t) for t in toks):
        hits_digit += 1

print(f"\ngazetteer (vendor-token exact match) coverage on {len(sample)} sample queries: {hits_brand_token} ({hits_brand_token/len(sample):.1%})")

# category-name token overlap (rough proxy for category gazetteer coverage)
cat_tokens = set()
for name in cat_names:
    for tok in re.findall(r"[а-яА-ЯёЁ]+", name.lower()):
        if len(tok) >= 4:
            cat_tokens.add(tok)

hits_cat_token = 0
for q in sample:
    toks = tokenize(q)
    if any(t in cat_tokens for t in toks):
        hits_cat_token += 1
print(f"category-name token overlap coverage: {hits_cat_token} ({hits_cat_token/len(sample):.1%})")

# query purity: for queries with >=5 click-rows, how concentrated is sku_subject_id / brand
print('\n--- query purity (category concentration) ---')
qc = df.groupby('query_text').size()
frequent_q = qc[qc >= 5].index
sub_df = df[df['query_text'].isin(frequent_q[:20000])]  # cap for speed
def top_share(s):
    vc = s.value_counts()
    return vc.iloc[0] / vc.sum()

purity_subject = sub_df.groupby('query_text')['sku_subject_id'].apply(top_share)
purity_brand = sub_df.groupby('query_text')['sku_brand_name'].apply(top_share)
print('subject_id purity (top category share of clicks per query):')
print(purity_subject.describe())
print(f"share of queries with >=90% clicks in one category: {(purity_subject>=0.9).mean():.1%}")
print(f"share of queries with <50% clicks in one category (ambiguous): {(purity_subject<0.5).mean():.1%}")

print('\nbrand purity (top brand share of clicks per query):')
print(purity_brand.describe())

# columns confirmation - no timestamp/session
print('\ncolumns available:', list(df.columns))
