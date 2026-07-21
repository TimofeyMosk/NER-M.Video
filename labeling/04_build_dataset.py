"""
Промт 4/4: финальная сборка train/val NER-датасета из трёх независимых источников.

Вход (если хоть одного нет — STOP, не регенерируем):
  labeling/output/weak_labels_clicks.parquet      (Промт 1)
  labeling/output/weak_labels_gazetteer.parquet   (Промт 2)
  labeling/output/synthetic_labels.parquet        (Промт 3)

Правила мёржа (заданы, не выдумываем):
  КАТЕГОРИЯ: clicks category_label при purity>=0.9 -> используем (source=clicks).
             иначе если словарь нашёл top-10 категорию -> source=gazetteer_only.
             иначе -> без категории, не в train.
  БРЕНД:     clicks-бренд и словарь согласуются -> high, используем.
             расходятся -> conflict=true -> golden_candidates, НЕ в train (весь пример).
             только один источник -> уверенность: purity>=0.9 high; 0.7-0.9 medium;
             только словарь medium; purity<0.7 без словаря = low (не в train, в golden).
  СИНТЕТИКА: отдельный помеченный пул (is_synthetic=true), не смешиваем её статистику с реальными.

Выравнивание спанов -> BIO:
  gazetteer-спаны уже позиционированы (берём напрямую);
  label-level значения (clicks/synthetic) ищем как подстроку; при неудаче — Левенштейн (малый порог);
  не нашли — сущность выбрасываем (позицию не выдумываем). Пример без сущностей -> drop.
  Бренд из кликов тегируется только если он РЕАЛЬНО в тексте (иначе majority-бренд не в запросе).

Сплит: GroupShuffleSplit по query_text 85/15 (один запрос не попадает и в train, и в val).

Артефакты:
  labeling/output/04_build_dataset.txt
  labeling/output/train.jsonl, val.jsonl           {"tokens":[...], "tags":[...]}
  labeling/output/golden_candidates.jsonl          {query_text, proposed_label, conflict}
"""
import os, sys, re, json, random, collections
import pandas as pd
import numpy as np
from Levenshtein import distance as lev
from sklearn.model_selection import GroupShuffleSplit

OUT_DIR = 'labeling/output'
F_CLICKS = os.path.join(OUT_DIR, 'weak_labels_clicks.parquet')
F_GAZ = os.path.join(OUT_DIR, 'weak_labels_gazetteer.parquet')
F_SYN = os.path.join(OUT_DIR, 'synthetic_labels.parquet')
F_ATTR = os.path.join(OUT_DIR, 'weak_labels_attr.parquet')  # типизированный ATTR (Промт 06)

TOP10 = ['Смартфоны', 'Стиральные машины', 'Холодильники', 'iPhone', 'Телевизоры',
         'Ноутбуки', 'Наушники', 'Пылесосы вертикальные', 'Аэрогрили', 'Планшеты']

SURFACE = {
    'Смартфоны': ['смартфон', 'смартфоны', 'телефон', 'телефоны'],
    'Стиральные машины': ['стиральная машина', 'стиральные машины', 'стиральная машинка', 'стиралка'],
    'Холодильники': ['холодильник', 'холодильники'],
    'iPhone': ['iphone', 'айфон', 'айфоны'],
    'Телевизоры': ['телевизор', 'телевизоры'],
    'Ноутбуки': ['ноутбук', 'ноутбуки', 'ноут'],
    'Наушники': ['наушники', 'наушник'],
    'Пылесосы вертикальные': ['вертикальный пылесос', 'пылесос вертикальный', 'пылесос', 'пылесосы'],
    'Аэрогрили': ['аэрогриль', 'аэрогрили'],
    'Планшеты': ['планшет', 'планшеты'],
}
# токены имён top-10 категорий (кириллица len>=4). Для gazetteer_only резолва требуем
# присутствие ВСЕХ токенов имени категории — иначе одиночный неоднозначный токен ('машины',
# 'вертикальные', 'пылесосы') спутает посудомоечную/швейную машину или пылесос-робот
# (не top-10) с top-10 категорией.
CAT_NAME_TOKENS = {c: {t for t in re.findall(r'[а-яё]+', c.lower()) if len(t) >= 4} for c in TOP10}

# латиница->кириллические варианты бренда (для выравнивания clicks-бренда, набранного кириллицей)
BRAND_TRANSLIT = {
    'apple': ['эпл', 'эппл', 'аппл'], 'samsung': ['самсунг'], 'xiaomi': ['сяоми', 'ксиоми'],
    'huawei': ['хуавей', 'хуавэй'], 'honor': ['хонор'], 'redmi': ['редми'], 'realme': ['реалми'],
    'poco': ['поко'], 'oppo': ['оппо'], 'vivo': ['виво'], 'tecno': ['техно'], 'bosch': ['бош'],
    'sony': ['сони'], 'philips': ['филипс'], 'dyson': ['дайсон'], 'haier': ['хайер', 'хаер'],
    'lg': ['лджи', 'элджи'], 'asus': ['асус'], 'lenovo': ['леново'], 'acer': ['эйсер'],
    'iphone': ['айфон'],
}

TOKEN_RE = re.compile(r'[0-9a-zа-яё]+')
PRIORITY = {'BRAND': 3, 'CATEGORY': 2, 'ATTR': 1}
# фильтр gazetteer-ATTR (имена параметров): выбрасываем предлоги-в-начале и мета-параметры,
# которые не являются осмысленными значениями характеристик для NER.
ATTR_STOP_FIRST = {'для', 'под', 'с', 'из', 'на', 'без', 'от', 'по', 'до', 'в', 'и', 'к'}
ATTR_STOP_FULL = {'страна', 'гарантия', 'гарантия предоставляется', 'модель', 'вес', 'ширина',
                  'высота', 'глубина', 'бренд', 'тип', 'размер', 'габаритные размеры', 'цвет',
                  'производитель', 'артикул', 'серия'}
random.seed(23)


def attr_ok(text):
    t = text.strip().lower()
    if t in ATTR_STOP_FULL:
        return False
    first = t.split()[0] if t.split() else t
    return first not in ATTR_STOP_FIRST


def tokenize(text):
    """список (token, start, end) по нижнему регистру."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text.lower())]


def lev_thr(n):
    return 0 if n <= 3 else (1 if n <= 6 else 2)


def align_surface(qtokens, surface):
    """Найти surface (возможно многословный) как окно токенов запроса (exact|Левенштейн).
    Возвращает (start_char, end_char) или None. qtokens: список (tok,s,e)."""
    sub = TOKEN_RE.findall(surface.lower())
    if not sub:
        return None
    n = len(sub)
    for i in range(len(qtokens) - n + 1):
        ok = True
        for j in range(n):
            qt = qtokens[i + j][0]
            st = sub[j]
            if qt == st:
                continue
            if lev(qt, st) <= lev_thr(min(len(qt), len(st))):
                continue
            ok = False
            break
        if ok:
            return (qtokens[i][1], qtokens[i + n - 1][2])
    return None


def align_first(qtokens, candidates):
    """Первый успешно выровненный кандидат (кандидаты уже отсортированы по приоритету)."""
    for c in candidates:
        sp = align_surface(qtokens, c)
        if sp:
            return sp
    return None


def to_bio(text, entities):
    """entities: список (type, start_char, end_char). BIO по токенам, приоритет типов."""
    qtokens = tokenize(text)
    tags = ['O'] * len(qtokens)
    owner = [None] * len(qtokens)  # (type, ent_id)
    for eid, (etype, s, e) in enumerate(entities):
        for ti, (tok, ts, te) in enumerate(qtokens):
            if ts < e and te > s:  # overlap
                cur = owner[ti]
                if cur is None or PRIORITY[etype] > PRIORITY[cur[0]]:
                    owner[ti] = (etype, eid)
    # проставить B/I по непрерывным блокам одного (type,eid)
    prev = None
    for ti in range(len(qtokens)):
        o = owner[ti]
        if o is None:
            tags[ti] = 'O'
            prev = None
        else:
            etype, eid = o
            if prev == (etype, eid):
                tags[ti] = 'I-' + etype
            else:
                tags[ti] = 'B-' + etype
            prev = (etype, eid)
    return [t[0] for t in qtokens], tags


def load_or_stop():
    for f in (F_CLICKS, F_GAZ, F_SYN, F_ATTR):
        if not os.path.exists(f):
            print(f"[STOP] нет входного файла: {f} — не регенерирую, останавливаюсь.")
            sys.exit(1)
    clicks = pd.read_parquet(F_CLICKS)
    syn = pd.read_parquet(F_SYN)
    return clicks, syn


def load_attr_lut():
    """query_text -> список типизированных ATTR-спанов {start,end,attr_type} (Промт 06).
    Спаны посчитаны на query_text.lower() — совпадают с офсетами to_bio(text.lower())."""
    attr = pd.read_parquet(F_ATTR)
    lut = {}
    for r in attr.itertuples(index=False):
        lut[r.query_text] = r.attr_span_matches
    return lut


def gaz_lookup(clicks_queries):
    """Строим lookup query_text -> (brand_spans, cat_spans, attr_spans) только для нужных запросов:
    (a) запросы из кликов, (b) запросы с top-10 категорийным токеном (для gazetteer_only)."""
    gaz = pd.read_parquet(F_GAZ)
    cq = set(clicks_queries)

    all_cat_tokens = set().union(*CAT_NAME_TOKENS.values())

    def has_top10_cat(spans):
        return any(s['text'] in all_cat_tokens for s in spans)

    in_clicks = gaz['query_text'].isin(cq)
    has_cat = gaz['category_span_matches'].apply(has_top10_cat)
    keep = gaz[in_clicks | has_cat]
    lut = {}
    for r in keep.itertuples(index=False):
        lut[r.query_text] = (r.brand_span_matches, r.category_span_matches, r.attr_span_matches)
    return lut, keep


def norm(s):
    return s.strip().lower() if isinstance(s, str) else s


def main():
    print("=== Промт 4/4: сборка train/val NER ===")
    clicks, syn = load_or_stop()
    print(f"[in] clicks={len(clicks):,} synthetic={len(syn):,}")
    lut, gaz_keep = gaz_lookup(clicks['query_text'].tolist())
    print(f"[in] gazetteer rows used (clicks∪top10-cat): {len(gaz_keep):,}")
    attr_lut = load_attr_lut()  # типизированный ATTR (Промт 06), заменяет param-имена
    print(f"[in] attr rows (typed): {len(attr_lut):,}")

    clicks_by_q = {r.query_text: r for r in clicks.itertuples(index=False)}
    clicks_qset = set(clicks_by_q)

    # gazetteer_only кандидаты на категорию: запросы НЕ из кликов, где присутствуют ВСЕ токены
    # имени ровно одной top-10 категории (полное имя категории в запросе).
    gaz_cat_only = []
    for q, (b, c, a) in lut.items():
        if q in clicks_qset:
            continue
        gtexts = {s['text'] for s in c}
        labels = [cat for cat, toks in CAT_NAME_TOKENS.items() if toks and toks <= gtexts]
        if len(labels) == 1:  # однозначная top-10 категория (полное имя присутствует)
            gaz_cat_only.append((q, labels[0]))

    stats = collections.Counter()
    train_rows = []       # dict: query_text, tokens, tags, category, brand, sources
    golden_conflict = []  # бренд-конфликт (самые спорные)
    golden_low = []       # low-confidence бренд

    def brand_gaz_norms(spans):
        out = set()
        for s in spans:
            out.add(norm(s.get('canon') or s['text']))
            out.add(norm(s['text']))
        return {x for x in out if x}

    def brand_gaz_span(spans, target_norm):
        for s in spans:
            if norm(s.get('canon') or s['text']) == target_norm or norm(s['text']) == target_norm:
                return (s['start'], s['end'])
        return None

    def process_real(q, cat_label, cat_source):
        """Возвращает train_row|None; побочно пишет в golden/stats."""
        gb, gc, ga = lut.get(q, ([], [], []))
        qtokens = tokenize(q)
        entities = []

        # --- CATEGORY span --- локализуем по surface-формам категории (морфология через Левенштейн).
        # Ярлык категории уже доверенный (клики purity>=0.9 / полное имя в словаре); нужно лишь
        # найти, ГДЕ в запросе упомянута категория. Одиночные gazetteer-токены не используем
        # для локализации, чтобы не привязать спан к чужому слову.
        cands = sorted(set(SURFACE.get(cat_label, []) + [cat_label.lower()]),
                       key=lambda x: -len(TOKEN_RE.findall(x)))
        cat_span = align_first(qtokens, cands)
        if cat_span:
            entities.append(('CATEGORY', cat_span[0], cat_span[1]))
        else:
            stats['cat_align_fail'] += 1

        # --- BRAND (правила мёржа) ---
        row = clicks_by_q.get(q)
        cb = norm(row.brand_label) if (row is not None and isinstance(row.brand_label, str)) else None
        cp = float(row.brand_purity) if (row is not None and row.brand_purity == row.brand_purity and cb) else None
        lit = bool(row.brand_literal_match) if row is not None else False
        gnorms = brand_gaz_norms(gb)

        brand_final, brand_conf, conflict = None, None, False
        if cb and gnorms:
            if cb in gnorms or any(cb in g or g in cb for g in gnorms):
                brand_final, brand_conf = cb, 'high'      # согласие
            else:
                conflict = True                            # расхождение -> golden, не train
        elif cb and not gnorms:
            brand_conf = 'high' if (cp is not None and cp >= 0.9) else ('medium' if (cp is not None and cp >= 0.7) else 'low')
            brand_final = cb
        elif gnorms and not cb:
            # только словарь: одиночный/латинский токен приоритетнее многословного
            # кириллического (последнее — часто категориеподобный мусор-vendor 'Смарт ТВ').
            brand_final = sorted(gnorms, key=lambda x: (x.count(' '), 0 if x.isascii() else 1, -len(x)))[0]
            brand_conf = 'medium'

        if conflict:
            stats['brand_conflict'] += 1
            golden_conflict.append({'query_text': q, 'proposed_label': f'BRAND clicks={cb} / gazetteer={sorted(gnorms)}',
                                    'conflict': True})
            return None  # весь пример -> golden, не в train

        if brand_final and brand_conf == 'low':
            stats['brand_low'] += 1
            golden_low.append({'query_text': q, 'proposed_label': f'BRAND?={brand_final} (purity<0.7, нет словаря)',
                               'conflict': False})
            brand_final = None  # low не тегируем в train

        # локализация бренда (только если реально в тексте)
        if brand_final:
            bspan = None
            if gnorms:  # согласие или gazetteer-only -> спан уже позиционирован
                bspan = brand_gaz_span(gb, brand_final)
            if bspan is None and lit:  # клики говорят: строка бренда есть в тексте
                cands = [brand_final] + BRAND_TRANSLIT.get(brand_final, [])
                bspan = align_first(qtokens, sorted(cands, key=lambda x: -len(x)))
            if bspan is None:  # попытка Левенштейна даже без literal_match
                cands = [brand_final] + BRAND_TRANSLIT.get(brand_final, [])
                bspan = align_first(qtokens, cands)
            if bspan:
                entities.append(('BRAND', bspan[0], bspan[1]))
            else:
                stats['brand_align_fail'] += 1

        # --- ATTR: типизированные значения из weak_labels_attr.parquet (Промт 06),
        #     уже позиционированы на q.lower(); в BIO кладём обобщённый тег ATTR ---
        for s in attr_lut.get(q, []):
            entities.append(('ATTR', s['start'], s['end']))

        if not entities:
            stats['dropped_no_entity'] += 1
            return None

        tokens, tags = to_bio(q, entities)
        if all(t == 'O' for t in tags):
            stats['dropped_no_entity'] += 1
            return None
        return {'query_text': q, 'tokens': tokens, 'tags': tags,
                'category': cat_label, 'brand': brand_final, 'source': cat_source}

    # --- реальные примеры: сначала клики (category source=clicks) ---
    for q in clicks_qset:
        row = clicks_by_q[q]
        r = process_real(q, row.category_label, 'clicks')
        if r:
            train_rows.append(r)
    # --- gazetteer_only категории ---
    for q, lab in gaz_cat_only:
        r = process_real(q, lab, 'gazetteer_only')
        if r:
            train_rows.append(r)

    real_df = pd.DataFrame(train_rows)
    # norm_key = нормализованный токен-текст. Токенизация снимает регистр/пунктуацию, поэтому
    # разные сырые query_text ("Холодильник"/"холодильник") дают один токен-текст. Группируем
    # сплит и дедуп ПО НЕМУ, иначе те же примеры утекают между train и val.
    real_df['norm_key'] = real_df['tokens'].apply(lambda t: ' '.join(t))
    before = len(real_df)
    real_df = real_df.drop_duplicates('norm_key').reset_index(drop=True)
    stats['real_dups_collapsed'] = before - len(real_df)
    stats['real_examples'] = len(real_df)

    # --- синтетика: отдельный пул, BIO той же схемой ---
    syn_rows = []
    for rr in syn.itertuples(index=False):
        q = rr.query_text
        qtokens = tokenize(q)
        ents = []
        if isinstance(rr.category_label, str):
            cands = sorted(set(SURFACE.get(rr.category_label, []) + [rr.category_label.lower()]),
                           key=lambda x: -len(TOKEN_RE.findall(x)))
            sp = align_first(qtokens, cands)
            if sp:
                ents.append(('CATEGORY', sp[0], sp[1]))
        if isinstance(rr.brand_label, str):
            sp = align_first(qtokens, [rr.brand_label.lower()] + BRAND_TRANSLIT.get(rr.brand_label.lower(), []))
            if sp:
                ents.append(('BRAND', sp[0], sp[1]))
        if isinstance(rr.attr_label, str):
            sp = align_surface(qtokens, rr.attr_label)
            if sp:
                ents.append(('ATTR', sp[0], sp[1]))
        if not ents:
            stats['syn_dropped'] += 1
            continue
        tokens, tags = to_bio(q, ents)
        syn_rows.append({'query_text': q, 'tokens': tokens, 'tags': tags, 'norm_key': ' '.join(tokens),
                         'category': rr.category_label, 'brand': rr.brand_label, 'source': 'synthetic'})
    syn_df = pd.DataFrame(syn_rows).drop_duplicates('norm_key').reset_index(drop=True)

    # --- golden: 500 «самых спорных» с квотами по трём типам (conflict / low / редкие категории) ---
    golden_rare = []
    if len(real_df):
        cat_counts = real_df['category'].value_counts()
        rare_cats = cat_counts[cat_counts < cat_counts.median() * 0.5].index.tolist()
        rare_pool = real_df[real_df['category'].isin(rare_cats)]
        take = rare_pool.sample(min(100, len(rare_pool)), random_state=1) if len(rare_pool) else rare_pool.head(0)
        for _, rr in take.iterrows():
            golden_rare.append({'query_text': rr['query_text'],
                                'proposed_label': f'CATEGORY={rr["category"]} (редкая категория)', 'conflict': False})

    def take_n(pool, n):
        return random.sample(pool, min(n, len(pool)))

    golden = take_n(golden_conflict, 250) + take_n(golden_low, 150) + golden_rare
    # добить до 500 из самого большого пула, если не хватило
    if len(golden) < 500:
        extra = take_n([g for g in golden_conflict + golden_low if g not in golden], 500 - len(golden))
        golden += extra
    golden = golden[:500]

    # --- GroupShuffleSplit по norm_key (85/15) на РЕАЛЬНЫХ; синтетику целиком в train ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, va_idx = next(gss.split(real_df, groups=real_df['norm_key']))
    train_real = real_df.iloc[tr_idx]
    val_real = real_df.iloc[va_idx]
    # синтетику только в train, но выбросить те синт.примеры, чей norm_key совпал с val (анти-утечка)
    val_keys = set(val_real['norm_key'])
    syn_train = syn_df[~syn_df['norm_key'].isin(val_keys)]
    stats['syn_dropped_leak'] = len(syn_df) - len(syn_train)
    train_all = pd.concat([train_real, syn_train], ignore_index=True)

    # проверка утечки по нормализованному токен-тексту (то, что реально видит модель)
    leak = set(train_all['norm_key']) & set(val_real['norm_key'])
    assert not leak, f"LEAK: {len(leak)} norm_key и в train и в val"

    # --- запись ---
    def dump(df, path):
        with open(path, 'w') as f:
            for _, r in df.iterrows():
                f.write(json.dumps({'tokens': r['tokens'], 'tags': r['tags']}, ensure_ascii=False) + '\n')

    dump(train_all, os.path.join(OUT_DIR, 'train.jsonl'))
    dump(val_real, os.path.join(OUT_DIR, 'val.jsonl'))
    with open(os.path.join(OUT_DIR, 'golden_candidates.jsonl'), 'w') as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')

    # ================= ОТЧЁТ =================
    print(f"\n=== размеры ===")
    print(f"train: {len(train_all):,}  (реальных {len(train_real):,} + синтетики {len(syn_train):,})")
    print(f"val:   {len(val_real):,}   (только реальные запросы)")
    print(f"golden_candidates: {len(golden):,}")
    print(f"дедуп реальных по norm_key: схлопнуто {stats['real_dups_collapsed']:,}; "
          f"синтетики выброшено как утечка в val: {stats['syn_dropped_leak']:,}")
    print(f"проверка утечки train↔val по norm_key: OK (пересечение = 0)")

    print(f"\n=== распределение по категориям в train (ТОЛЬКО реальные, без синтетики) ===")
    print(train_real['category'].value_counts().reindex(TOP10).to_string())

    print(f"\n=== топ-15 брендов в train (только реальные) ===")
    print(train_real[train_real['brand'].notna()]['brand'].value_counts().head(15).to_string())

    total_real_cand = len(clicks_qset) + len(gaz_cat_only)
    print(f"\n=== качество мёржа (знаменатель = {total_real_cand:,} реальных кандидатов: "
          f"{len(clicks_qset):,} clicks + {len(gaz_cat_only):,} gazetteer_only) ===")
    print(f"оставлено в реальный пул: {len(real_df):,}")
    print(f"% примеров с conflict=true (бренд) -> golden, не в train: "
          f"{stats['brand_conflict']}/{total_real_cand} = {stats['brand_conflict']/max(1,total_real_cand):.1%}")
    print(f"% выброшенных из-за неудачного alignment (нет ни одной сущности): "
          f"{stats['dropped_no_entity']}/{total_real_cand} = {stats['dropped_no_entity']/max(1,total_real_cand):.1%}")
    print(f"  не удалось локализовать категорию (спан категории не проставлен): {stats['cat_align_fail']:,}")
    print(f"  не удалось локализовать бренд (спан бренда не проставлен): {stats['brand_align_fail']:,}")
    print(f"low-confidence брендов -> golden (бренд снят, пример остаётся ради категории): {stats['brand_low']:,}")
    print(f"синтетики выброшено (не выровнялось): {stats['syn_dropped']:,}")

    print(f"\n=== 10 случайных примеров из train (tokens / tags) ===")
    for _, r in train_all.sample(10, random_state=7).iterrows():
        pairs = ' '.join(f'{t}/{g}' for t, g in zip(r['tokens'], r['tags']))
        print(f"   {pairs}")

    print(f"\n[saved] train.jsonl, val.jsonl, golden_candidates.jsonl в {OUT_DIR}/")


if __name__ == '__main__':
    main()
