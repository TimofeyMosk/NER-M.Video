import pickle, json, collections

with open('cu_ws/skus.pkl','rb') as f:
    obj = pickle.load(f)

shop = obj['yml_catalog']['shop']
cats = shop['categories']['category']
offers = shop['offers']['offer']

# category tree stats
cat_ids = {c['@id'] for c in cats}
root_cats = [c for c in cats if '@parentId' not in c]
print(f"Categories total: {len(cats)}, root categories: {len(root_cats)}")

# offers stats
n = len(offers)
print(f"Offers total: {n}")

key_counts = collections.Counter()
avail_counts = collections.Counter()
has_vendor = 0
has_model = 0
param_name_counts = collections.Counter()
param_name_search_counts = collections.Counter()
n_params_dist = collections.Counter()
cat_id_counts = collections.Counter()
vendor_counts = collections.Counter()
missing_name = 0
name_lengths = []
multi_category_offers = 0

for o in offers:
    for k in o.keys():
        key_counts[k] += 1
    avail_counts[o.get('@available','?')] += 1
    if o.get('vendor'):
        has_vendor += 1
        vendor_counts[o['vendor']] += 1
    if o.get('model'):
        has_model += 1
    name = o.get('name')
    if not name:
        missing_name += 1
    else:
        name_lengths.append(len(name))
    catobj = o.get('categories')
    if catobj:
        cid = catobj.get('categoryId')
        if isinstance(cid, list):
            multi_category_offers += 1
            for c in cid:
                cat_id_counts[c] += 1
        else:
            cat_id_counts[cid] += 1
    params = o.get('param', [])
    if isinstance(params, dict):
        params = [params]
    n_params_dist[len(params)] += 1
    for p in params:
        pname = p.get('@name','?')
        param_name_counts[pname] += 1
        if pname.endswith('_search'):
            param_name_search_counts[pname[:-7]] += 1

print("\n--- offer keys presence ---")
for k,v in key_counts.most_common():
    print(f"{k}: {v} ({v/n:.1%})")

print("\n--- availability ---")
print(avail_counts)

print(f"\nhas_vendor: {has_vendor} ({has_vendor/n:.1%})")
print(f"has_model: {has_model} ({has_model/n:.1%})")
print(f"missing_name: {missing_name}")
print(f"multi_category_offers: {multi_category_offers}")
print(f"unique category ids referenced by offers: {len(cat_id_counts)}")
print(f"unique vendors: {len(vendor_counts)}")

print("\n--- name length stats (chars) ---")
import statistics
print(f"min={min(name_lengths)} max={max(name_lengths)} mean={statistics.mean(name_lengths):.1f} median={statistics.median(name_lengths)}")

print("\n--- params per offer distribution (percentiles) ---")
all_np = []
for k,v in n_params_dist.items():
    all_np.extend([k]*v)
all_np.sort()
import numpy as np
arr = np.array(all_np)
print(f"min={arr.min()} p25={np.percentile(arr,25)} median={np.median(arr)} p75={np.percentile(arr,75)} p95={np.percentile(arr,95)} max={arr.max()}")

print(f"\nunique param names (raw): {len(param_name_counts)}")
print(f"unique '_search' fact names: {len(param_name_search_counts)}")

print("\n--- top 30 most common '_search' fact names ---")
for name, cnt in param_name_search_counts.most_common(30):
    print(f"{cnt:>7}  {name}")

print("\n--- top 20 most common NON-search param names ---")
non_search = collections.Counter({k:v for k,v in param_name_counts.items() if not k.endswith('_search')})
for name, cnt in non_search.most_common(20):
    print(f"{cnt:>7}  {name}")

print("\n--- top 20 vendors by offer count ---")
for name, cnt in vendor_counts.most_common(20):
    print(f"{cnt:>7}  {name}")

print("\n--- top 20 categories by offer count ---")
cat_id_to_name = {c['@id']: c['#text'] for c in cats}
for cid, cnt in cat_id_counts.most_common(20):
    print(f"{cnt:>7}  {cid}  {cat_id_to_name.get(cid,'?')}")

# save param name lists for later use
import os
os.makedirs('eda/output', exist_ok=True)
with open('eda/output/search_param_names.json','w') as f:
    json.dump(param_name_search_counts.most_common(), f, ensure_ascii=False, indent=1)

with open('eda/output/vendor_names.json','w') as f:
    json.dump(vendor_counts.most_common(), f, ensure_ascii=False, indent=1)

with open('eda/output/category_names.json','w') as f:
    json.dump([{'id':c['@id'],'parent':c.get('@parentId'),'name':c['#text']} for c in cats], f, ensure_ascii=False, indent=1)
