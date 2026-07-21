"""
Промт 3/4: НЕЗАВИСИМЫЙ источник — синтетические запросы из структуры каталога skus.pkl.

Ни клики, ни словарные совпадения не используются. Берём реальные комбинации из каталога:
  категория+бренд  — vendor офферов данной категории
  категория+атрибут — «чистые» атрибуты: Страна (Страна_search), Цвет (Цвет_search, split по '/').
Форма (Форма_search) в топ-10 категориях ОТСУТСТВУЕТ (это атрибут мебели/матрасов) — не используется.

Шаблоны (телеграфные, как реальные поисковые запросы, lowercase):
  "{category} {brand}"            напр. "холодильник haier"
  "{brand} {category}"            напр. "samsung телевизор"      (частый порядок у пользователей)
  "{category} {attr_value}"       напр. "наушники черные", "стиралка россия"
  "{brand} {category} {attr_value}" напр. "xiaomi пылесос белый"
Категории подставляются в НАТУРАЛЬНЫХ формах (смартфон/телефон, стиралка, тв, ноут, айфон),
а не как формальные ярлыки каталога, чтобы синтетика звучала правдоподобно.

Артефакты:
  labeling/output/03_synthetic_labels.txt     — отчёт
  labeling/output/synthetic_labels.parquet     — датасет (query_text, category_label, brand_label, attr_label, is_synthetic)
"""
import os, pickle, random, collections
import pandas as pd

SKUS = 'cu_ws/skus.pkl'
OUT_DIR = 'labeling/output'
OUT_PARQUET = os.path.join(OUT_DIR, 'synthetic_labels.parquet')

TOP10 = ['Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
         'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты']

# натуральные формы категорий (как пишут пользователи); category_label остаётся каноничным из TOP10
SURFACE = {
    'Смартфоны': ['смартфон', 'телефон'],
    'Стиральные машины': ['стиральная машина', 'стиралка', 'стиральная машинка'],
    'Холодильники': ['холодильник'],
    'iPhone': ['iphone', 'айфон'],
    'Телевизоры': ['телевизор', 'тв'],
    'Ноутбуки': ['ноутбук', 'ноут'],
    'Наушники': ['наушники'],
    'Пылесосы вертикальные': ['вертикальный пылесос', 'пылесос вертикальный', 'пылесос'],
    'Аэрогрили': ['аэрогриль'],
    'Планшеты': ['планшет'],
}

COUNTRY_FIX = {'респ.корея': 'корея'}
COUNTRY_DROP = {'евросоюз', 'гонконг', 'оаэ'}  # звучат неестественно в запросе

# лимиты на разнообразие (реальные значения, топ по частоте в категории)
N_BRANDS = 30       # топ брендов на категорию
N_COLORS = 22       # топ цветов
N_COUNTRIES = 6     # топ стран
N_T3_BRANDS = 30    # бренды для тройного шаблона
N_T3_COLORS = 18
CAP_PER_CAT = 1500  # потолок примеров на категорию (синтетика <=20% train)

random.seed(17)


def split_color(v):
    return [x.strip().lower() for x in str(v).split('/') if x.strip()]


def collect():
    obj = pickle.load(open(SKUS, 'rb'))
    shop = obj['yml_catalog']['shop']
    id2name = {c['@id']: c['#text'] for c in shop['categories']['category']}
    per = {c: {'brand': collections.Counter(), 'brand_case': {},
               'country': collections.Counter(), 'color': collections.Counter(),
               'forma': collections.Counter()} for c in TOP10}
    for o in shop['offers']['offer']:
        c = o.get('categories')
        nm = id2name.get(c.get('categoryId')) if isinstance(c, dict) else None
        if nm not in TOP10:
            continue
        v = o.get('vendor')
        if v and v.strip().lower() not in ('нет бренда', 'без бренда', 'no name'):
            vl = v.strip().lower()
            per[nm]['brand'][vl] += 1
            per[nm]['brand_case'].setdefault(vl, collections.Counter())[v.strip()] += 1
        ps = o.get('param', [])
        if isinstance(ps, dict):
            ps = [ps]
        for p in ps:
            n, t = p.get('@name'), p.get('#text')
            if not t:
                continue
            if n == 'Страна_search':
                cv = str(t).strip().lower()
                cv = COUNTRY_FIX.get(cv, cv)
                if cv not in COUNTRY_DROP:
                    per[nm]['country'][cv] += 1
            elif n == 'Цвет_search':
                for cc in split_color(t):
                    per[nm]['color'][cc] += 1
            elif n == 'Форма_search':
                per[nm]['forma'][str(t).strip().lower()] += 1
    return per


def canon_brand(cat_data, brand_lower):
    """Каноничное написание бренда (самое частое в каталоге)."""
    cc = cat_data['brand_case'].get(brand_lower)
    return cc.most_common(1)[0][0] if cc else brand_lower


def gen():
    per = collect()
    rows = []  # (query_text, category_label, brand_label, attr_label)
    forma_total = sum(sum(per[c]['forma'].values()) for c in TOP10)

    for cat in TOP10:
        d = per[cat]
        surfaces = SURFACE[cat]
        brands = [b for b, _ in d['brand'].most_common(N_BRANDS)]
        colors = [c for c, _ in d['color'].most_common(N_COLORS)]
        countries = [c for c, _ in d['country'].most_common(N_COUNTRIES)]
        attrs = [('color', v) for v in colors] + [('country', v) for v in countries]

        cat_rows = []

        def surf():
            return random.choice(surfaces)

        # T1: {category} {brand}  и  T2order: {brand} {category}
        for b in brands:
            bl_label = canon_brand(d, b)
            cat_rows.append((f"{surf()} {b}", cat, bl_label, None))
            cat_rows.append((f"{b} {surf()}", cat, bl_label, None))
        # T3: {category} {attr}
        for atype, av in attrs:
            cat_rows.append((f"{surf()} {av}", cat, None, av))
        # T4: {brand} {category} {attr}
        t3_brands = brands[:N_T3_BRANDS]
        t3_colors = colors[:N_T3_COLORS]
        for b in t3_brands:
            bl_label = canon_brand(d, b)
            for av in t3_colors:
                cat_rows.append((f"{b} {surf()} {av}", cat, bl_label, av))
            for av in countries:
                cat_rows.append((f"{b} {surf()} {av}", cat, bl_label, av))

        # dedup по query_text внутри категории, затем потолок
        seen = set()
        uniq = []
        random.shuffle(cat_rows)
        for r in cat_rows:
            if r[0] in seen:
                continue
            seen.add(r[0])
            uniq.append(r)
        if len(uniq) > CAP_PER_CAT:
            uniq = random.sample(uniq, CAP_PER_CAT)
        rows.extend(uniq)

    return rows, per, forma_total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Промт 3/4: синтетика из каталога (независимо от кликов/словарей) ===")

    rows, per, forma_total = gen()
    df = pd.DataFrame(rows, columns=['query_text', 'category_label', 'brand_label', 'attr_label'])
    df = df.drop_duplicates('query_text').reset_index(drop=True)
    df['is_synthetic'] = True

    print(f"\nФорма_search в топ-10 категориях: {forma_total} значений -> атрибут неприменим к скоупу, не используется")
    print(f"использованы чистые атрибуты: Страна, Цвет (split по '/')")

    print(f"\nвсего сгенерировано: {len(df):,} запросов")
    print("\nпо категориям (кол-во / из них с брендом / с атрибутом):")
    g = df.groupby('category_label')
    breakdown = pd.DataFrame({
        'n': g.size(),
        'with_brand': g['brand_label'].apply(lambda s: s.notna().sum()),
        'with_attr': g['attr_label'].apply(lambda s: s.notna().sum()),
    }).reindex(TOP10)
    print(breakdown.to_string())

    print("\n10 случайных примеров (оценить на глаз естественность):")
    for _, r in df.sample(10, random_state=3).iterrows():
        print(f"   {r['query_text']!r:45}  cat={r['category_label']}  brand={r['brand_label']}  attr={r['attr_label']}")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[saved] {OUT_PARQUET}  ({len(df):,} rows, is_synthetic=True)")


if __name__ == '__main__':
    main()
