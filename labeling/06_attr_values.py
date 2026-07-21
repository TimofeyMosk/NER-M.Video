"""
Извлечение ТИПИЗИРОВАННЫХ ATTR-фактов из запросов по ЗНАЧЕНИЯМ каталога (замена param-name подхода).

Механизмы (precision-first):
  1. value-gazetteer  — закрытые словари значений из caталога per-category:
       color, country, resolution, energy_class, fridge_type, headphone_type,
       vacuum_type, control_type, os, cpu  (split по '/', фильтр по длине/цифрам).
  2. feature-лексикон — курируемые фичи-флаги: 5g/wifi/oled/qled/nfc/esim/смарт/инверторный/...
  3. numeric+unit     — regex «число+единица» (гб/см/дюйм/гц/мач/л/вт) -> тип по единице (high).
  4. numeric bare     — голое число, ТОЛЬКО если оно ∈ множеству значений каталога для этой
                        категории (память смартфона/планшета; диагональ ТВ/планшета/ноута).
                        Так отсекаются модель-коды («iphone 15» -> 15∉set -> не тег).

Guard от загрязнения: значение НЕ тегируется, если совпадает с брендом или категорийным словом.

Вход:  weak_labels_clicks.parquet (query_text + category_label — категория нужна, чтобы выбрать
        правильный словарь значений). skus.pkl — источник значений.
Выход: weak_labels_attr.parquet: query_text, category_label,
        attr_span_matches=[{text, attr_type, value, start, end, method, confidence}]

Запуск: .venv/bin/python3 labeling/06_attr_values.py [N]   (N = размер выборки; без арг — полный)
"""
import os, sys, re, json, pickle, collections
import pandas as pd
import ahocorasick

OUT_DIR = 'labeling/output'
CLICKS = os.path.join(OUT_DIR, 'weak_labels_clicks.parquet')
SKUS = 'cu_ws/skus.pkl'
VENDORS = 'timofey/eda/output/vendor_names.json'
OUT_PARQUET = os.path.join(OUT_DIR, 'weak_labels_attr.parquet')

TOP10 = ['Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
         'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты']

# param каталога -> тип атрибута (только «чистые» словарные)
PARAM_ATTR = {
    'Цвет_search': 'color', 'Страна_search': 'country',
    'Класс энергоэффективности_search': 'energy_class',
    'Тип холодильника_search': 'fridge_type', 'Тип наушников_search': 'headphone_type',
    'Тип пылесоса_search': 'vacuum_type', 'Тип управления_search': 'control_type',
    'Операционная система_search': 'os', 'Модель процессора_search': 'cpu',
    'Акустический тип наушников_search': 'headphone_ac_type',
}
# числовые значения из каталога (для bare-number матчинга)
PARAM_NUMERIC = {'Встроенная память (ROM)_search': 'memory', 'Диагональ_search': 'diagonal'}
BARE_MEMORY_CATS = {'Смартфоны', 'Планшеты', 'iPhone'}
BARE_DIAGONAL_CATS = {'Телевизоры', 'Планшеты', 'Ноутбуки'}

# курируемые словари (то, что в каталоге грязно/многословно, но частотно в запросах)
CURATED = {
    'resolution': ['4k', '8k', 'uhd', 'qhd', 'full hd', 'fullhd', 'hd ready'],
    'cpu': ['core i3', 'core i5', 'core i7', 'core i9', 'ryzen 3', 'ryzen 5', 'ryzen 7',
            'ryzen 9', 'snapdragon', 'celeron', 'pentium', 'm1', 'm2', 'm3', 'm4', 'm5'],
    'fridge_type': ['side-by-side', 'двухкамерный', 'двухдверный', 'однокамерный', 'однодверный',
                    'многодверный', 'комби', 'french door'],
}
FEATURES = {  # surface -> (attr_type, value)
    '5g': ('network', '5g'), '4g': ('network', '4g'), 'lte': ('network', 'lte'),
    'wifi': ('connectivity', 'wifi'), 'wi-fi': ('connectivity', 'wifi'),
    'nfc': ('connectivity', 'nfc'), 'bluetooth': ('connectivity', 'bluetooth'),
    'esim': ('sim', 'esim'), 'nano-sim': ('sim', 'nano-sim'),
    'oled': ('panel', 'oled'), 'qled': ('panel', 'qled'), 'amoled': ('panel', 'amoled'),
    'ips': ('panel', 'ips'), 'mini led': ('panel', 'mini led'),
    'смарт': ('smart', 'smart'), 'smart': ('smart', 'smart'),
    'инверторный': ('motor', 'инверторный'), 'инверторная': ('motor', 'инверторный'),
    'инвертор': ('motor', 'инверторный'),
    'с сушкой': ('feature', 'с сушкой'), 'с паром': ('feature', 'с паром'),
    'сенсорный': ('control_type', 'сенсорный'), 'сенсорное': ('control_type', 'сенсорный'),
}
UNIT_ATTR = {  # единица -> тип (для regex «число+единица»)
    'гб': 'memory', 'gb': 'memory', 'тб': 'memory', 'tb': 'memory',
    'дюйм': 'diagonal', 'дюйма': 'diagonal', 'дюймов': 'diagonal',
    'см': 'dimension', 'мм': 'dimension',
    'гц': 'refresh', 'hz': 'refresh', 'герц': 'refresh', 'мач': 'battery', 'mah': 'battery',
    'л': 'volume', 'литр': 'volume', 'литра': 'volume', 'литров': 'volume',
    'вт': 'power', 'квт': 'power',
}
# lookbehind: число НЕ приклеено к букве (иначе это модель-код, напр. F2V9GW9W -> «9вт»).
# 'w' как единицу мощности не берём — слишком часто внутри модель-кодов.
NUM_UNIT_RE = re.compile(r'(?<![0-9a-zа-яё])(\d+(?:[.,]\d+)?)\s*(гб|gb|тб|tb|дюйм\w*|см|мм|герц\w*|гц|hz|мач|mah|литр\w*|л|квт|вт)\b')
TOKEN_RE = re.compile(r'[0-9a-zа-яё]+')
WORDCH_RE = re.compile(r'[0-9a-zа-яё]')

sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def is_wordchar(ch):
    return bool(WORDCH_RE.match(ch))


def build_dicts():
    obj = pickle.load(open(SKUS, 'rb'))
    shop = obj['yml_catalog']['shop']
    id2name = {c['@id']: c['#text'] for c in shop['categories']['category']}
    vendors = {v[0].strip().lower() for v in json.load(open(VENDORS))}
    # категорийные слова (guard от загрязнения)
    cat_words = set()
    for c in TOP10:
        for t in re.findall(r'[а-яё]+', c.lower()):
            if len(t) >= 4:
                cat_words.add(t)
    cat_words |= {'смартфон', 'телефон', 'холодильник', 'телевизор', 'ноутбук', 'наушники',
                  'планшет', 'пылесос', 'аэрогриль', 'машина', 'машинка', 'айфон', 'iphone',
                  'вертикальный', 'вертикальная', 'вертикальные', 'стиральная', 'стиральные'}

    per_val = collections.defaultdict(lambda: collections.Counter())  # cat -> {(surface): attr}
    per_val_attr = collections.defaultdict(dict)
    per_num = collections.defaultdict(lambda: collections.defaultdict(set))  # cat -> attr -> {int values}

    def add_val(cat, surface, attr):
        s = surface.strip().lower()
        if not (2 <= len(s) <= 30):
            return
        if len(s.split()) > 3:
            return
        if re.search(r'\d', s):  # значения с цифрами -> в числовые, не сюда
            return
        if s in vendors or s in cat_words:  # guard от загрязнения
            return
        if s in ('да', 'нет', 'есть'):
            return
        per_val_attr[cat][s] = attr

    for o in shop['offers']['offer']:
        c = o.get('categories')
        nm = id2name.get(c.get('categoryId')) if isinstance(c, dict) else None
        if nm not in TOP10:
            continue
        ps = o.get('param', [])
        ps = [ps] if isinstance(ps, dict) else ps
        for p in ps:
            n, t = p.get('@name'), p.get('#text')
            if not t:
                continue
            if n in PARAM_ATTR:
                attr = PARAM_ATTR[n]
                for piece in str(t).split('/'):
                    v = piece.strip().lower()
                    if attr == 'energy_class' and '+' not in v:  # только a+/a++/a+++
                        continue
                    if attr == 'cpu' and re.search(r'\d\.\d|ггц|ghz', v):  # без частот
                        v = re.sub(r'\s*\d+(?:[.,]\d+)?\s*(ггц|ghz).*', '', v).strip()
                    add_val(nm, v, attr)
            elif n in PARAM_NUMERIC:
                attr = PARAM_NUMERIC[n]
                m = re.match(r'(\d+)', str(t).strip())
                if m:
                    per_num[nm][attr].add(int(m.group(1)))
    # курируемые
    for attr, surfs in CURATED.items():
        for cat in TOP10:
            for s in surfs:
                per_val_attr[cat][s] = attr
    for cat in TOP10:
        for s, (attr, val) in FEATURES.items():
            per_val_attr[cat][s] = attr  # value восстановим по FEATURES при выводе
    return per_val_attr, per_num, vendors, cat_words


def build_automata(per_val_attr):
    autos = {}
    for cat, d in per_val_attr.items():
        A = ahocorasick.Automaton()
        for surface, attr in d.items():
            A.add_word(surface, (attr, len(surface), surface))
        if len(A):
            A.make_automaton()
        autos[cat] = A
    return autos


def scan_values(ql, automaton):
    hits = []
    for end_idx, (attr, klen, surf) in automaton.iter(ql):
        s = end_idx - klen + 1
        e = end_idx + 1
        left = s == 0 or not is_wordchar(ql[s - 1])
        right = e == len(ql) or not is_wordchar(ql[e])
        if left and right:
            val = FEATURES[surf][1] if surf in FEATURES else surf
            hits.append({'text': ql[s:e], 'attr_type': attr, 'value': val,
                         'start': s, 'end': e, 'method': 'feature' if surf in FEATURES else 'gazetteer_value',
                         'confidence': 'high'})
    return hits


def scan_numeric(ql, cat, per_num):
    hits = []
    covered = []
    # (a) число + единица
    for m in NUM_UNIT_RE.finditer(ql):
        num, unit = m.group(1), m.group(2)
        unit_key = unit if unit in UNIT_ATTR else unit.rstrip('аов')  # дюймов/литров -> базовая
        attr = UNIT_ATTR.get(unit) or UNIT_ATTR.get(unit[:4]) or UNIT_ATTR.get(unit[:2])
        if not attr:
            continue
        hits.append({'text': m.group(0).strip(), 'attr_type': attr, 'value': f'{num} {unit}',
                     'start': m.start(), 'end': m.end(), 'method': 'numeric_unit', 'confidence': 'high'})
        covered.append((m.start(), m.end()))
    # (b) голое число ∈ множеству значений каталога
    mem = per_num.get(cat, {}).get('memory', set())
    dia = per_num.get(cat, {}).get('diagonal', set())
    for m in re.finditer(r'\d+', ql):
        s, e = m.start(), m.end()
        if any(cs <= s < ce for cs, ce in covered):
            continue
        # стоять отдельным словом (не приклеено к буквам -> не модель-код)
        if (s > 0 and is_wordchar(ql[s - 1])) or (e < len(ql) and is_wordchar(ql[e])):
            continue
        num = int(m.group())
        attr = None
        if cat in BARE_MEMORY_CATS and num in mem and num >= 32:
            attr = 'memory'
        elif cat in BARE_DIAGONAL_CATS and num in dia:
            attr = 'diagonal'
        if attr:
            hits.append({'text': m.group(), 'attr_type': attr, 'value': str(num),
                         'start': s, 'end': e, 'method': 'numeric_bare', 'confidence': 'medium'})
    return hits


def dedup(spans):
    spans = sorted(spans, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    kept = []
    for sp in spans:
        if any(o['start'] <= sp['start'] and sp['end'] <= o['end'] and o is not sp and
               (o['end'] - o['start']) > (sp['end'] - sp['start']) for o in spans):
            continue
        if any(k['start'] == sp['start'] and k['end'] == sp['end'] for k in kept):
            continue
        kept.append(sp)
    return kept


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 06: типизированный ATTR из значений каталога ===")
    print(f"mode: {'SAMPLE ' + str(sample_n) if sample_n else 'FULL'}")

    per_val_attr, per_num, vendors, cat_words = build_dicts()
    autos = build_automata(per_val_attr)
    print("значений в словарях per-category:", {c: len(per_val_attr[c]) for c in TOP10})
    print("числовых множеств:", {c: {a: len(v) for a, v in per_num.get(c, {}).items()} for c in TOP10 if per_num.get(c)})

    df = pd.read_parquet(CLICKS, columns=['query_text', 'category_label'])
    if sample_n:
        df = df.sample(min(sample_n, len(df)), random_state=42)
    print(f"запросов на вход: {len(df):,}")

    rows = []
    for q, cat in zip(df['query_text'], df['category_label']):
        ql = q.lower()
        A = autos.get(cat)
        spans = scan_values(ql, A) if A is not None and len(A) else []
        spans += scan_numeric(ql, cat, per_num)
        spans = dedup(spans)
        rows.append((q, cat, spans))

    out = pd.DataFrame({'query_text': [r[0] for r in rows],
                        'category_label': [r[1] for r in rows],
                        'attr_span_matches': [r[2] for r in rows]})
    matched = out[out['attr_span_matches'].str.len() > 0].copy()

    print(f"\n=== покрытие ===")
    print(f"запросов с >=1 ATTR: {len(matched):,} / {len(out):,} = {len(matched)/len(out):.1%}")
    tc = collections.Counter(s['attr_type'] for r in rows for s in r[2])
    mc = collections.Counter(s['method'] for r in rows for s in r[2])
    print("\nтипы атрибутов (кол-во спанов):")
    for t, c in tc.most_common():
        print(f"  {t:16} {c:,}")
    print("методы:", dict(mc))

    print("\n15 примеров:")
    for _, r in matched.sample(min(15, len(matched)), random_state=3).iterrows():
        facts = [f"{s['attr_type']}={s['value']}" for s in r['attr_span_matches']]
        print(f"   [{r['category_label']:12}] {r['query_text']!r:48} -> {facts}")

    if not sample_n:
        matched.to_parquet(OUT_PARQUET, index=False)
        print(f"\n[saved] {OUT_PARQUET}  ({len(matched):,} rows)")
    else:
        print("\n[sample mode] parquet НЕ записан")


if __name__ == '__main__':
    main()
