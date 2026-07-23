# Repository file summaries

This inventory started from the 57 original files and was updated on 2026-07-22. The authoritative deployment inventory is now `PRODUCTION_DEPLOYMENT.md` plus the machine-readable `production_manifest.json`.

## Production runtime additions

- `unified_query_annotator.py` — Public `annotate_query` API and interactive/JSON CLI combining normalization, stopwords, dictionaries, measurements, colors and partial BIO.
- `requirements-runtime.txt` — Minimal online dependencies; ETL and Transformer libraries are intentionally excluded.
- `production_manifest.json` — Exact 14-file online deployment contract.
- `verify_production_runtime.py` — Completeness, smoke-case, span/BIO and throughput verification.
- `search_dictionaries/output/brands.json`, `categories.json`, `category_alias_overrides.json`, `color_aliases.csv` — The four compiled artifacts read by online inference.
- `.github/workflows/production-check.yml` — Clean Python 3.12 CI using runtime-only dependencies.

## Root

- `.gitattributes` — Marks all processed Parquet outputs under `pavel_nntp/data/processed` as Git LFS-managed binary files.
- `.gitignore` — Standard broad Python ignore list plus project-specific exclusions for `.DS_Store`, `cu_ws/`, and `cu_ws.zip`; raw source datasets are intentionally not committed.
- `README.md` — Introduces the M.Video intelligent e-commerce search case: extract category, brand, and product attributes from query text and return structured JSON in under 100 ms using Python or Go.
- `CLAUDE.md` — Expanded project brief and technical notes. Clarifies that the model consumes search queries, not product cards; records EDA findings, weak-label risks, an MVP plan, and the expected Python-training/Go-inference architecture.
- `FILE_SUMMARIES.md` — This generated file-by-file inventory.

## Go inference stub

- `inference/.DS_Store` — macOS Finder metadata accidentally committed; no application logic.
- `inference/internal/.DS_Store` — Another macOS Finder metadata file; no application logic.
- `inference/go.mod` — Defines the minimal Go module `inference`, targeting Go 1.26.2, with no dependencies.
- `inference/cmd/main.go` — Empty `package main` declaration. The production inference service has not been implemented yet.

## `pavel_nntp`: pipeline and analysis

- `pavel_nntp/README.md` — Navigation guide for Pavel's work: explains the pipeline scripts, smoke test, processed-data layout, source-data expectations, and recommended starting points.
- `pavel_nntp/data_processing_pipeline.md` — Detailed design specification for the complete query-click processing pipeline: schema, normalization, aliases, catalog joins, duplicate handling, position weighting, query-level aggregation, confidence scoring, filtering, splitting, validation, outputs, and reproducibility.
- `pavel_nntp/data_processing_pipeline_final.md` — Short final version of the same design, organized as ten execution stages from loading through train/validation output.
- `pavel_nntp/process_query_clicks.py` — Main ETL/weak-label pipeline. Extracts catalog entities and attributes, normalizes and canonicalizes queries, separates positive/no-click rows, collapses duplicates, joins catalog data, computes evidence/confidence, aggregates query labels, rejects poor records, creates grouped stratified train/valid splits, and writes reports, dictionaries, config, hashes, and manifest.
- `pavel_nntp/export_normalization_sample.py` — Deterministically samples 1,000 distinct raw queries, applies the pipeline normalizer, and exports side-by-side CSV and HTML review files; also refreshes the manifest.
- `pavel_nntp/query_clicks_report.py` — Pandas-based generator for a standalone Russian HTML EDA report covering data quality, duplicates, top queries/brands/categories/SKUs, positions, prices, and query-text length.
- `pavel_nntp/query_clicks_analysis.html` — Generated visual EDA report from the previous script, with sections for schema quality, duplicates, popular queries, brands/categories, position bias, prices, text statistics, and top SKUs.
- `pavel_nntp/test_process_query_clicks_smoke.py` — End-to-end smoke test using temporary synthetic Parquet inputs. Redirects pipeline outputs to a temp directory and checks normalization, aggregation, filtering, and split behavior without touching production artifacts.
- `pavel_nntp/inspect_train.ipynb` — Generic experiment notebook scaffold. It currently demonstrates seeded random-number summary code and contains placeholders for hypothesis, metrics, results, and next steps; it does not yet inspect the real train dataset.
- `pavel_nntp/Мвидео преза.pdf` — M.Video presentation PDF associated with the case. Its exact slide contents were not directly extractable in this runtime; it is likely the original case/background deck rather than executable project material.

## `pavel_nntp/data`: configuration and metadata

- `pavel_nntp/data/README.md` — Describes the generated `processed`, `dictionaries`, `config`, and `reports` subdirectories and the role of each artifact.
- `pavel_nntp/data/config/processing_config.yaml` — Reproducible pipeline settings: seed 42, source/output paths, NFKC/lowercase/ё-to-е/dash normalization, protected symbols, Russian-to-canonical aliases, position-0 no-click rule, evidence formula, confidence and length thresholds, and an 80/20 grouped stratified split.
- `pavel_nntp/data/manifest.json` — Build manifest dated 2026-07-21. Records the generating script, dataset metrics, output file sizes, SHA-256 hashes, and validation facts such as 30,991,350 raw rows, 1,789,579 unique queries, 982,761 eligible queries, 786,891 train and 195,870 validation queries, and zero split overlap.

## Dictionaries

- `pavel_nntp/data/dictionaries/attribute_aliases.json` — Empty object reserved for future alternate spellings of attribute names.
- `pavel_nntp/data/dictionaries/attribute_names.json` — Canonical vocabulary of 10,171 searchable catalog attribute names extracted from `_search` parameters.
- `pavel_nntp/data/dictionaries/brand_aliases.json` — Reviewed map of 47 observed Cyrillic/colloquial brand surfaces to canonical Latin brand names.
- `pavel_nntp/data/dictionaries/brands.json` — Canonical catalog brand dictionary containing 6,479 normalized brands used for query matching and weak labels.
- `pavel_nntp/data/dictionaries/categories.json` — Full 6,963-node product-category dictionary/tree extracted from the catalog.
- `pavel_nntp/data/dictionaries/category_aliases.json` — Reviewed Russian/English category synonyms, including broad canonical mappings such as phone/iPhone to smartphones.
- `pavel_nntp/data/dictionaries/product_family_aliases.json` — Maps Russian product-family spellings such as `айфон`, `айпад`, and `макбук` to canonical names.
- `pavel_nntp/data/dictionaries/product_family_to_brand.json` — Connects canonical product families such as iPhone/iPad/MacBook to their parent brands, enabling implicit brand inference.
- `pavel_nntp/data/dictionaries/selected_attribute_values.json` — Curated value vocabularies for five relatively closed attributes: material, body material, country, color, and shape; intended as gazetteers for MVP extraction.

## Processed Parquet datasets

- `pavel_nntp/data/processed/catalog_skus.parquet` — Normalized SKU-level catalog table for 546,069 offers, including identifiers, names, brands, and category information used in joins and label resolution.
- `pavel_nntp/data/processed/catalog_attributes.parquet` — Long-form table of 14,548,468 searchable SKU attribute rows, covering 10,171 attribute names; basis for attribute dictionaries and weak labels.
- `pavel_nntp/data/processed/query_sku_aggregated.parquet` — Largest intermediate output. Aggregates positive click evidence by canonical query–SKU pair after duplicate collapse and position weighting; the manifest reports 3,460,618 query–SKU pairs.
- `pavel_nntp/data/processed/query_dataset.parquet` — Query-level weakly labeled dataset, aggregating clicked SKUs into dominant category/brand/entities, confidence, quality, and stratification fields; 1,010,293 aggregated queries before final eligibility filtering.
- `pavel_nntp/data/processed/no_click_queries.parquet` — Distinct/aggregated queries represented only by position-0 rows, used as negative or diagnostic data; 456,233 queries according to the manifest.
- `pavel_nntp/data/processed/rejected_rows.parquet` — 602,164 excluded records with rejection reasons such as low confidence, excessive length, missing evidence, or other quality rules.
- `pavel_nntp/data/processed/train.parquet` — Final training split of 786,891 canonical queries, grouped to avoid query leakage and stratified by category, frequency, length, entities, and quality.
- `pavel_nntp/data/processed/valid.parquet` — Final validation split of 195,870 canonical queries (about 19.93%), constructed with the same grouping/stratification and zero canonical-query overlap with train.

## Generated pipeline reports

- `pavel_nntp/data/reports/data_processing_report.html` — Pipeline QA dashboard showing quality tiers, rejection reasons, top categories and brands, and train/validation category distributions.
- `pavel_nntp/data/reports/query_normalization_1000.csv` — Auditable two-column sample of 1,000 original queries and normalized outputs; demonstrates lowercasing and normalization while leaving unknown typos unchanged.
- `pavel_nntp/data/reports/query_normalization_1000.html` — Human-friendly HTML rendering of the same 1,000 normalization pairs.

## `timofey/eda`: exploratory scripts and report

- `timofey/eda/prep_prompt.md` — A structured Russian prompt requesting a one-hour theory primer for the case: NER/query understanding, weak supervision, normalization, taxonomy/attributes, metrics, inference, and an actionable MVP plan.
- `timofey/eda/01_skus_catalog.py` — Loads the pickled Yandex Market-style catalog and profiles categories, offers, availability, vendors/models, parameter cardinality, `_search` attributes, category counts, and offer completeness; exports lookup JSON and text logs.
- `timofey/eda/02_query_clicks.py` — Loads the 31-million-row click log and profiles nulls, uniqueness, full/pair duplicates, query frequencies and lengths, position bias, prices, brands, category IDs, sample rows, and SKU overlap with catalog/descriptions.
- `timofey/eda/03_query_clicks_extra.py` — Investigates why `sku_subject_id` cannot be treated as the catalog category ID, examines position-0 rows, measures Cyrillic/Latin/digit composition and case-only duplicates, and estimates literal clicked-brand presence in queries.
- `timofey/eda/04_mvp_signals.py` — Tests MVP-friendly signals: value-set sizes for selected attributes, brand/category gazetteer coverage, and query-level category/brand purity. Finds very high category concentration but weaker brand concentration.
- `timofey/eda/eda_report.html` — Consolidated visual EDA report for the three original datasets, covering overlap, click/query distributions, text complications, resolved categories, catalog characteristics, risks, and next-day recommendations.

## `timofey/eda/output`: generated EDA artifacts

- `timofey/eda/output/01_skus_catalog.txt` — Console report from catalog profiling: 6,963 categories, 546,069 offers, vendor/parameter coverage, category leaders, and structural details of the YML-like catalog.
- `timofey/eda/output/02_query_clicks.txt` — Detailed click-log statistics: 30,991,350 rows, 1,789,579 unique queries, 332,753 clicked SKUs, 40% exact duplicates, strong top-position bias, query-length/price distributions, top queries/brands/categories, and cross-dataset overlap.
- `timofey/eda/output/03_query_clicks_extra.txt` — Supplemental diagnostics: only 1 of the top 20 subject IDs maps directly to a catalog category, 23.8% of rows have position 0, 58.7% of queries contain digits, 39.7% mix Latin and Cyrillic, and only about 27% literally contain the clicked SKU's brand.
- `timofey/eda/output/04_mvp_signals.txt` — MVP signal results: brand gazetteer coverage 57.2%, category-name token overlap 44.9%, 93% of sampled queries have at least 90% of clicks in one category, and only 1% are strongly category-ambiguous.
- `timofey/eda/output/brand_counts.parquet` — Compact frequency table of catalog brands/vendors, used to rank and prioritize brand vocabulary.
- `timofey/eda/output/category_names.json` — Lookup of all 6,963 catalog category IDs to human-readable names for resolving/reporting category results.
- `timofey/eda/output/overlap_stats.json` — Seven cross-source overlap metrics: 332,753 clicked SKUs, 546,069 catalog SKUs, 1,177,200 description SKUs, plus pairwise and three-way intersections (102,945 in all three).
- `timofey/eda/output/search_param_names.json` — List of 10,184 raw `_search` parameter names discovered in catalog offers; slightly larger than the final normalized attribute-name dictionary.
- `timofey/eda/output/unique_queries.parquet` — Deduplicated query table derived from the click log, containing roughly 1.79 million unique query strings for text analysis and sampling.
- `timofey/eda/output/vendor_names.json` — List of 6,765 unique raw catalog vendor names used for brand gazetteer coverage experiments; normalization/deduplication later reduces this to 6,479 canonical brands.

## Overall state

The repository now contains the reproducible data pipeline, silver validation, RuBERT evaluation utilities and a working Python production annotator. Its minimal online package is 14 files (about 6.73 MB) and does not require raw Parquet/pickle data or ETL/Transformer libraries. The Go API remains a scaffold. The main modeling risks remain noisy click labels, position bias, snapshot mismatch, mixed-script/model-code text and the large category-dependent attribute space.
