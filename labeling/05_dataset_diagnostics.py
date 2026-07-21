"""
Диагностика итогового train/val датасета (для оценки готовности к дообучению rubert-tiny2).
Не меняет данные — только считает метрики риска. Вывод: labeling/output/05_dataset_diagnostics.txt
"""
import json, collections, re
import numpy as np

OUT = 'labeling/output'
TOKEN_RE = re.compile(r'[0-9a-zа-яё]+')


def load(p):
    return [json.loads(l) for l in open(p)]


def spans(tags):
    """список (type) сущностей по BIO."""
    ents = []
    for t in tags:
        if t.startswith('B-'):
            ents.append(t[2:])
    return ents


def main():
    tr = load(f'{OUT}/train.jsonl')
    va = load(f'{OUT}/val.jsonl')
    print(f"train={len(tr):,}  val={len(va):,}")

    # --- длины запросов ---
    lens = [len(r['tokens']) for r in tr]
    print(f"\nдлина (токенов): median={int(np.median(lens))} p95={int(np.percentile(lens,95))} max={max(lens)}")

    # --- распределение тегов ---
    tagc = collections.Counter(t for r in tr for t in r['tags'])
    tot = sum(tagc.values())
    print("\nраспределение тегов (train):")
    for t, c in tagc.most_common():
        print(f"  {t:12} {c:>8,}  {c/tot:.1%}")

    # --- по сущностям: сколько примеров имеют каждую комбинацию ---
    combo = collections.Counter()
    per_ex_ent = collections.Counter()
    for r in tr:
        e = set(spans(r['tags']))
        combo[tuple(sorted(e)) or ('<none>',)] += 1
        for x in e:
            per_ex_ent[x] += 1
    print("\nсколько примеров содержат сущность (train):")
    for k in ('CATEGORY', 'BRAND', 'ATTR'):
        print(f"  {k}: {per_ex_ent[k]:,} ({per_ex_ent[k]/len(tr):.1%})")
    print("комбинации сущностей в примере:")
    for k, c in combo.most_common(8):
        print(f"  {'+'.join(k):22} {c:>7,} ({c/len(tr):.1%})")

    # --- дубли / почти-дубли ---
    texts = [' '.join(r['tokens']) for r in tr]
    dup = len(texts) - len(set(texts))
    print(f"\nполные дубли по тексту токенов (train): {dup:,} ({dup/len(tr):.1%})")

    # --- val: можно ли на нём мерить brand/attr F1? ---
    va_ent = collections.Counter()
    for r in va:
        for x in set(spans(r['tags'])):
            va_ent[x] += 1
    print(f"\nval: примеров с сущностью — "
          f"CATEGORY {va_ent['CATEGORY']:,} ({va_ent['CATEGORY']/len(va):.1%}), "
          f"BRAND {va_ent['BRAND']:,} ({va_ent['BRAND']/len(va):.1%}), "
          f"ATTR {va_ent['ATTR']:,} ({va_ent['ATTR']/len(va):.1%})")

    # --- баланс брендов: сколько уникальных, хвост ---
    br = collections.Counter()
    for r in tr:
        toks, tags = r['tokens'], r['tags']
        cur = []
        for tk, tg in zip(toks, tags):
            if tg == 'B-BRAND':
                if cur: br[' '.join(cur)] += 1
                cur = [tk]
            elif tg == 'I-BRAND':
                cur.append(tk)
            else:
                if cur: br[' '.join(cur)] += 1
                cur = []
        if cur: br[' '.join(cur)] += 1
    print(f"\nуникальных brand-поверхностей в train: {len(br):,}; "
          f"встречаются 1 раз: {sum(1 for v in br.values() if v==1):,}")

    # --- ATTR: что реально попало (топ поверхностей) ---
    at = collections.Counter()
    for r in tr:
        toks, tags = r['tokens'], r['tags']
        cur = []
        for tk, tg in zip(toks, tags):
            if tg == 'B-ATTR':
                if cur: at[' '.join(cur)] += 1
                cur = [tk]
            elif tg == 'I-ATTR':
                cur.append(tk)
            else:
                if cur: at[' '.join(cur)] += 1
                cur = []
        if cur: at[' '.join(cur)] += 1
    print(f"\nуникальных ATTR-поверхностей: {len(at):,}. Топ-20:")
    print("  " + ", ".join(f"{k}×{v}" for k, v in at.most_common(20)))

    # --- оценка «категория без бренда» / «бренд без категории» ---
    cat_only = combo[('CATEGORY',)]
    brand_only = combo[('BRAND',)]
    print(f"\nтолько категория: {cat_only:,} ({cat_only/len(tr):.1%}); "
          f"только бренд: {brand_only:,} ({brand_only/len(tr):.1%})")

    # --- РИСК 1: утечка train<->val по нормализованному тексту (сплит был по сырому query_text) ---
    tr_txt = set(texts)
    leak = sum(1 for r in va if ' '.join(r['tokens']) in tr_txt)
    print(f"\n[РИСК] утечка train<->val (одинаковый токен-текст): {leak:,}/{len(va):,} = {leak/len(va):.1%}")

    # --- РИСК 2: загрязнение ATTR категорийными/брендовыми словами ---
    CATW = {'смартфон', 'смартфоны', 'телефон', 'холодильник', 'холодильники', 'телевизор',
            'телевизоры', 'ноутбук', 'ноутбуки', 'наушники', 'планшет', 'планшеты', 'пылесос',
            'пылесосы', 'аэрогриль', 'машина', 'машинка', 'стиральная', 'iphone', 'айфон'}
    brands = {b[0].lower() for b in json.load(open('timofey/eda/output/vendor_names.json'))}
    tot_attr = sum(at.values())
    cat_as_attr = sum(v for k, v in at.items() if k in CATW)
    brand_as_attr = sum(v for k, v in at.items() if k in brands)
    print(f"[РИСК] ATTR = категорийное слово: {cat_as_attr:,} ({cat_as_attr/tot_attr:.1%}); "
          f"ATTR = бренд: {brand_as_attr:,} ({brand_as_attr/tot_attr:.1%})")


if __name__ == '__main__':
    main()
