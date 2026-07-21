"""
Оценка предсказаний по GOLD-сету (честная приёмка). Считает entity-level P/R/F1 (seqeval) по типам,
плюс разрез «словарь поймал (gazetteer_hit) / промахнулся» — чтобы видеть маргинальную пользу модели.

Работает без модели: по умолчанию сравнивает ЭТАЛОН (gold.jsonl) с словарным baseline
(gold_review.jsonl — черновые теги нашего словаря/правил). Позже можно подать предсказания модели
через --pred.

Оба файла — jsonl со строками {query_text, tokens, tags, ...}. Джойн по query_text; токены должны
совпадать (иначе строка пропускается с предупреждением).

Запуск:
  # baseline словаря против эталона:
  .venv/bin/python3 labeling/08_eval_gold.py --gold labeling/output/gold.jsonl
  # качество модели:
  .venv/bin/python3 labeling/08_eval_gold.py --gold labeling/output/gold.jsonl --pred model_pred.jsonl
"""
import os, sys, json, argparse, collections
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

OUT_DIR = 'labeling/output'


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def index_by_query(rows):
    return {r['query_text']: r for r in rows}


def collect(gold_rows, pred_by_q):
    """Возвращает (y_true, y_pred, meta_hit[]) по совпадающим запросам с совпадающими токенами."""
    y_true, y_pred, hit = [], [], []
    skipped = 0
    for g in gold_rows:
        p = pred_by_q.get(g['query_text'])
        if p is None or p['tokens'] != g['tokens']:
            skipped += 1
            continue
        y_true.append(g['tags'])
        y_pred.append(p['tags'])
        hit.append(bool(g.get('meta', {}).get('gazetteer_hit', False)))
    return y_true, y_pred, hit, skipped


def report(y_true, y_pred, title):
    if not y_true:
        print(f"\n[{title}] нет данных")
        return
    print(f"\n=== {title}  (n={len(y_true)}) ===")
    print(f"overall  P={precision_score(y_true, y_pred):.3f} "
          f"R={recall_score(y_true, y_pred):.3f} F1={f1_score(y_true, y_pred):.3f}")
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for k, v in rep.items():
        if isinstance(v, dict) and not k.endswith('avg'):
            print(f"  {k:10} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1-score']:.3f} (n={int(v['support'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default=os.path.join(OUT_DIR, 'gold.jsonl'))
    ap.add_argument('--pred', default=os.path.join(OUT_DIR, 'gold_review.jsonl'),
                    help='предсказания (по умолчанию — словарный черновик gold_review.jsonl)')
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        print(f"[STOP] нет эталона {args.gold}. Сначала разметьте gold_review.jsonl -> gold.jsonl "
              f"(см. labeling/GOLD_GUIDELINES.md).")
        sys.exit(1)

    gold = load(args.gold)
    reviewed = [g for g in gold if g.get('meta', {}).get('reviewed')]
    if reviewed:
        print(f"[info] размечено (reviewed=true): {len(reviewed)}/{len(gold)}")
        gold = reviewed
    else:
        print(f"[warn] ни одна строка не помечена meta.reviewed=true — оцениваю все {len(gold)} как есть")

    pred_by_q = index_by_query(load(args.pred))
    print(f"[in] gold={len(gold)}  pred={args.pred} ({len(pred_by_q)} строк)")

    y_true, y_pred, hit, skipped = collect(gold, pred_by_q)
    if skipped:
        print(f"[warn] пропущено {skipped} (нет в pred или не совпали токены)")

    report(y_true, y_pred, "ВСЕ запросы")
    # разрез по покрытию словарём
    for flag, name in [(True, "словарь ПОЙМАЛ (gazetteer_hit=true)"),
                       (False, "словарь ПРОМАХНУЛСЯ (gazetteer_hit=false — территория модели)")]:
        yt = [t for t, h in zip(y_true, hit) if h == flag]
        yp = [p for p, h in zip(y_pred, hit) if h == flag]
        report(yt, yp, name)

    print("\nПодсказка: сравните F1 на «промахнулся» для словаря vs модели — там видно, "
          "что реально добавляет обученная модель.")


if __name__ == '__main__':
    main()
