# Обучение rubert-tiny2 (NER) — переносимый пакет

Дообучение `cointegrated/rubert-tiny2` как token-classifier (BIO: CATEGORY / BRAND / ATTR)
для query-understanding поиска М.Видео. Пакет самодостаточен — данные лежат внутри,
сырьё `cu_ws/` на машине обучения **не нужно**.

## Содержимое

```
requirements.txt   зависимости (torch + transformers + seqeval; без accelerate/datasets)
labels.json        7 меток BIO в фиксированном порядке (id ↔ label)
train_ner.py       обучение + оценка (seqeval) + замер латентности
predict.py         инференс обученной модели по jsonl -> предсказанные BIO (для оценки на gold)
data/train.jsonl   96k примеров  {"tokens":[...], "tags":[...]}
data/val.jsonl     16k примеров  (чистый held-out, без утечки — сплит по норм. токен-тексту)
```

Данные уже нормализованы (lowercase), сплит train/val сделан по нормализованному токен-тексту
(пересечение train↔val = 0), ATTR — типизированные значения из каталога.

## Запуск на GPU-машине

```bash
# 1) torch под вашу CUDA (пример для CUDA 12.1):
pip install torch>=2.3 --index-url https://download.pytorch.org/whl/cu121
# 2) остальное:
pip install -r requirements.txt
# 3) обучение (CUDA + fp16 включаются автоматически):
python train_ner.py
```

Быстрый смоук-тест (проверить, что всё запускается):

```bash
python train_ner.py --max-train 500 --epochs 1
```

Полезные флаги: `--epochs`, `--batch-size`, `--lr`, `--max-length` (по умолчанию 64), `--no-cuda`.

## Результат

```
model/                    сохранённая лучшая (по val entity-F1) модель + токенизатор
model/metrics.json        per-type precision/recall/F1 на val + p50/p95 латентности (CPU, batch=1)
```

В логе печатаются per-epoch val P/R/F1, финальный отчёт по типам сущностей и латентность.

## Честная оценка на gold-сете (после обучения)

`val` размечен теми же weak-правилами, поэтому меряет согласие с ними, а не истину. Для честной
приёмки — эталонный gold (`labeling/output/gold.jsonl`, размечается вручную по
`labeling/GOLD_GUIDELINES.md`). Прогнать модель по нему и оценить:

```bash
python predict.py                 # model/ -> labeling/output/model_pred.jsonl
# из корня репозитория:
python labeling/08_eval_gold.py --gold labeling/output/gold.jsonl --pred labeling/output/model_pred.jsonl
```

`08_eval_gold.py` печатает entity-level P/R/F1 по типам + разрез «словарь поймал / промахнулся» —
там видно, что именно добавляет обученная модель поверх словарного baseline.

## Важно знать

- **Метрики — против weak-разметки** (val размечен теми же правилами), т.е. это согласие с
  weak-лейблами, а не абсолютная истина. Для честной приёмки нужен отдельный gold-сет
  (кандидаты в `labeling/output/golden_candidates.jsonl`).
- **ATTR** в тегах обобщённый (`B/I-ATTR`); тип атрибута (память/цвет/диагональ/…) в проде
  восстанавливается детерминированным слоем (см. `labeling/06_attr_values.py`).
- **Латентность** меряется на CPU batch=1 как ориентир под приёмку <100мс; финальный прод-замер —
  на целевом CPU и (опц.) квантованной ONNX-модели.
- Токенизатор WordPiece сохраняется в `model/` — он же понадобится для паритета при Go-инференсе.
