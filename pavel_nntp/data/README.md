# Обработанные данные

Папка создана скриптом `pavel_nntp/process_query_clicks.py`.

## processed

- `catalog_skus.parquet` — нормализованный каталог и связь SKU с категориями.
- `catalog_attributes.parquet` — индексируемые `_search` характеристики товаров.
- `query_sku_aggregated.parquet` — агрегированные положительные связи запрос–SKU.
- `query_dataset.parquet` — итоговая таблица на уровне канонического запроса.
- `no_click_queries.parquet` — запросы, для которых встречался `sku_position = 0`.
- `rejected_rows.parquet` — исключенные записи с причиной исключения.
- `train.parquet` — обучающая выборка.
- `valid.parquet` — валидационная выборка.

## dictionaries

Ручные алиасы, категории, бренды и словари характеристик в JSON. Автоматические и fuzzy-алиасы не используются.

## config

`processing_config.yaml` содержит все правила и пороги обработки.

## reports

`data_processing_report.html` содержит контрольные статистики и распределения.

`manifest.json` хранит метрики запуска, размеры файлов и SHA-256.
