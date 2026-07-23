# NER-M.Video

## Production quick start

Минимальный production-набор, точный список файлов и правила загрузки в GitHub описаны в [`PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md). Комплект также зафиксирован машинно в [`production_manifest.json`](production_manifest.json).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe verify_production_runtime.py
.\.venv\Scripts\python.exe unified_query_annotator.py
```

Для online-аннотации не требуются DuckDB, pandas, PyArrow, Torch, Transformers и исходные parquet/pickle.

## Полное окружение разработки

Production-комплект проверен в локальном окружении `.venv` с Python 3.12. Чтобы результат не зависел от системного Python и глобальных пакетов, VS Code должен использовать `${workspaceFolder}\\.venv\\Scripts\\python.exe`.

```powershell
cd C:\Users\kozgu\NER-M.Video
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

В VS Code выполните `Python: Select Interpreter` → `.venv\\Scripts\\python.exe` и откройте новый терминал. Локальная папка `.vscode` не загружается в GitHub.

## Единая предобработка и BIO-разметка

Основная runtime-точка входа — `unified_query_annotator.py`. Она объединяет нормализацию, стоп-слова и intent-флаги, словари брендов/явных категорий, measurement parser, палитру цветов и единый positive-only BIO. Контролируемые синонимы и product families приводятся к общим категориям: например, `телефон/айфон/iphone → Смартфоны`, `макбук → Ноутбуки`, `airpods → Наушники`.

```powershell
# Ручной интерактивный ввод
.venv\Scripts\python.exe unified_query_annotator.py

# Один запрос
.venv\Scripts\python.exe unified_query_annotator.py "телевизор Samsung черный 55 дюймов"

# JSONL-файл с результатами
.venv\Scripts\python.exe unified_query_annotator.py --json `
  --output output/manual_annotations.jsonl `
  "монитор LG серый 27 дюймов 144 Гц"
```

Python API:

```python
from unified_query_annotator import annotate_query

annotation = annotate_query("ноутбук HP серый 16 ГБ")
model_text = annotation["texts"]["query_model_input"]
```

Подробная схема и BIO-контракт описаны в [`PIPELINE_HANDOFF.md`](PIPELINE_HANDOFF.md), а production-доставка — в [`PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md).

## Материалы проекта

- [Комплексный аудит экспериментов](experiments/EXPERIMENT_AUDIT_RU.md)
- [План следующих экспериментов](experiments/NEXT_EXPERIMENTS_RU.md)
- [Реестр и протокол экспериментов](experiments/README.md)
- [Предобработка текстов и HTML-отчёт](text_preprocessing/README.md)
- [Silver-разметка 15% корпуса для validation](silver_annotation/README.md)
- [Evaluator RuBERT на silver-validation и полном корпусе](rubert_evaluation/README.md)
- [Поисковые словари брендов, категорий и единиц измерения](search_dictionaries/README.md)
- [Production-файлы, deployment и GitHub checklist](PRODUCTION_DEPLOYMENT.md)

Кейс с которым мы будем работать:
Интеллектуальный поиск для e-commerce (МВидео)
NLP, NER, Python/Go, Поисковые системы

ML

Описание кейса
Поисковая система М.Видео должна определять категорию, бренды и характеристики товара, чтобы точно отвечать на запросы пользователей. Точность выдачи напрямую влияет на конверсию в покупку. 

Команде предстоит создать алгоритм, который будет правильно извлекать сущности из поисковых запросов.

Требования к кандидатам
Базовые знания в Data Science и NLP

Владение языком программирования Python или Go

Понимание трансформерных архитектур

Результаты буткемпа
Ты научишься:

Работать с NER (технологией распознавания именованных сущностей) и поисковыми алгоритмами

Создавать сервисы

Презентовать решения стейкхолдерам

Зарегистрироваться на буткемп

Ответы на частые вопросы
Что должно получиться в итоге? 
Словари, модели или сервисы на Python или Go.

Что считается успешным результатом? 
Алгоритм или сервис, который получает «запрос» и менее чем за 100 мс выдает структурированный ответ с фактами в формате JSON.
