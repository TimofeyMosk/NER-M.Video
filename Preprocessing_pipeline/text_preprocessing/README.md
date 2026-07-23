# Предобработка поисковых запросов

Пайплайн реализует требуемые операции без регулярных выражений:

1. точная дедупликация буквально исходных строк;
2. Unicode NFKC с сохранением исходной строки в отдельной колонке;
3. замена `ё` на `е`;
4. lowercase;
5. замена дефисов и кавычек пробелом;
6. нормализация пробелов;
7. безопасная русская лемматизация;
8. повторное `ё → е`, нормализация пробелов и дедупликация по итоговому тексту.

До удаления кавычек и дефисов параллельно строится `query_model`: минимально очищенный текст для BIO NER. На нём запускаются `MeasurementParser` и `ColorNormalizer`, которые сохраняют символьные spans и положительные BIO-кандидаты характеристик и цветов.

Лемматизация выполняется после технической нормализации. Иначе дефисные конструкции воспринимаются как один токен, а варианты регистра и `ё/е` зря увеличивают морфологический словарь.

## Защита товарных сущностей

Не лемматизируются:

- токены с латиницей;
- токены с цифрами;
- неизвестные морфологическому словарю слова;
- однословные бренды из каталога.

Это снижает риск повреждения брендов, моделей и артикулов. Результат остаётся baseline-ом: контекстная морфологическая неоднозначность без модели не разрешается.

## Запуск

```powershell
.venv\Scripts\python.exe text_preprocessing\preprocess_queries.py
```

Быстрый контрольный запуск:

```powershell
.venv\Scripts\python.exe text_preprocessing\preprocess_queries.py `
  --limit 5000 `
  --output-dir text_preprocessing\output_sample
```

Тесты:

```powershell
.venv\Scripts\python.exe -m unittest text_preprocessing\test_preprocess_queries.py
.venv\Scripts\python.exe text_preprocessing\verify_outputs.py
.venv\Scripts\python.exe -m unittest search_dictionaries.test_measurement_parser
.venv\Scripts\python.exe text_preprocessing\verify_measurement_preprocessing.py
.venv\Scripts\python.exe -m unittest search_dictionaries.test_color_normalizer
.venv\Scripts\python.exe text_preprocessing\verify_color_preprocessing.py
```

## Measurement parser и BIO

Parser использует 108 типов единиц и token trie без регулярных выражений. Он формирует два уровня:

- `measurement_candidates_json` — все допустимые кандидаты числа и единицы;
- `measurement_bio_entities_json` — только кандидаты, для которых контекст позволяет выбрать тип `screen_diagonal`, `refresh_rate`, `battery_capacity`, `power`, `memory_ram` или `memory_rom`.

BIO-разметка является положительной и частичной: токены вне подтверждённых spans получают `O`, но `measurement_bio_mask_json=false`. Их нельзя использовать как размеченные отрицательные примеры.

`ГБ` без контекста RAM/ROM, единицы без числа, однобуквенные алиасы, `M2`, `pixel`, модельные коды и конфликт `HP` не превращаются в BIO-факты автоматически.

## Нормализация цветов и BIO

`ColorNormalizer` сводит каталожные оттенки к фиксированной палитре из 30 классов. Он использует longest-match token trie без регулярных выражений и учитывает русские словоформы, английские названия, каталожные составные значения и ручные безопасные алиасы.

Примеры:

- `темно-синий` / `navy` → `синий`;
- `антрацит` / `space gray` → `серый`;
- `слоновая кость` → `кремовый`;
- `rose gold` → `золотистый`;
- `черный/красный` → `разноцветный`.

Исходный `query_model` не переписывается: offsets `color_bio_entities_json` относятся именно к нему. Отдельная колонка `query_color_canonical` предназначена для cache key, классификатора или дополнительного входа модели и не должна использоваться для восстановления исходных spans.

Как и measurement-разметка, цветовая BIO-разметка является положительной и частичной. Брендовые фразы защищены отдельным trie: например, `black` внутри известного бренда не становится цветом. Неуверенные маркетинговые названия не применяются автоматически.

## Результаты

- `output/preprocessed_queries_audit.parquet` — строка на исходный уникальный запрос, все промежуточные стадии, `query_model`, measurement/color spans, tokens, BIO-теги и supervision mask.
- `output/preprocessed_queries.parquet` — итоговые уникальные тексты, число исходных вариантов и до пяти примеров.
- `output/preprocessing_metrics.json` — метрики преобразований.
- `output/preprocessing_samples.csv` — компактная таблица примеров.
- `output/text_preprocessing_report.html` — автономный HTML-лендинг с графиками и выводами.

Крупные Parquet-файлы локальны и исключены из Git в `text_preprocessing/output/.gitignore`.

Пустая финальная строка не попадает в `preprocessed_queries.parquet`, но остаётся в audit-файле для разбора причины. В cache-представлении кавычки удаляются, но MeasurementParser уже успевает обработать обозначение дюймов в `query_model`.
