from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from pavel_nntp import process_query_clicks as pipeline
except ModuleNotFoundError:
    import process_query_clicks as pipeline


def write_table(path: Path, rows: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows), path, compression="zstd")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        pipeline.QUERY_CLICKS_PATH = root / "query_clicks.parquet"
        pipeline.QUERY_MAPPING_PATH = root / "query_mapping.parquet"
        pipeline.CATALOG_SKUS_PATH = root / "catalog_skus.parquet"
        pipeline.QUERY_SKU_PATH = root / "query_sku.parquet"
        pipeline.NO_CLICK_PATH = root / "no_click.parquet"
        pipeline.QUERY_DATASET_PATH = root / "query_dataset.parquet"
        pipeline.REJECTED_PATH = root / "rejected.parquet"
        pipeline.TRAIN_PATH = root / "train.parquet"
        pipeline.VALID_PATH = root / "valid.parquet"
        pipeline.REPORT_PATH = root / "report.html"
        pipeline.WORK_DIR = root / "work"
        pipeline.WORK_DIR.mkdir()

        queries = [
            "Айфон-15 Pro",
            "Iphone 15 Pro",
            "Самсунг S24",
            "Холодильник",
            "Холодильники",
            "Телевизор",
            "Телевизоры",
            "Ноутбук",
            "Ноутбуки",
            "Планшет",
        ]
        raw_queries = [
            queries[0], queries[0], queries[1], queries[2], queries[2], queries[2],
            queries[3], queries[3], queries[4], queries[5], queries[5], queries[6],
            queries[7], queries[7], queries[8], queries[9], queries[9], queries[2],
        ]
        sku_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, None]
        positions = [1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 0]
        write_table(
            pipeline.QUERY_CLICKS_PATH,
            {
                "sku_id": sku_ids,
                "toValidUTF8(sku_name)": ["Товар"] * len(raw_queries),
                "toValidUTF8(sku_brand_name)": [""] * len(raw_queries),
                "sku_price": [100.0] * len(raw_queries),
                "sku_subject_id": [10] * len(raw_queries),
                "sku_seo_id": [20] * len(raw_queries),
                "toValidUTF8(query_text)": raw_queries,
                "sku_position": positions,
            },
        )

        mapping_rows = {
            "query_original": queries,
            "query_normalized": [pipeline.normalize_text(value) for value in queries],
            "query_canonical": [pipeline.canonicalize_text(pipeline.normalize_text(value)) for value in queries],
            "query_length_chars": [len(pipeline.normalize_text(value)) for value in queries],
            "query_length_words": [len(pipeline.normalize_text(value).split()) for value in queries],
            "has_digits": [any(char.isdigit() for char in value) for value in queries],
            "has_latin": [False] * len(queries),
            "has_cyrillic": [True] * len(queries),
            "is_mixed_language": [False] * len(queries),
            "has_dictionary_brand": [False, False, True, False, False, False, False, False, False, False],
            "has_dictionary_product_family": [True, False, False, False, False, False, False, False, False, False],
        }
        write_table(pipeline.QUERY_MAPPING_PATH, mapping_rows)
        write_table(
            pipeline.CATALOG_SKUS_PATH,
            {
                "sku_id": [1, 2, 3, 4, 5, 6],
                "catalog_sku_name": ["iPhone", "Samsung", "Холодильник", "Телевизор", "Ноутбук", "Планшет"],
                "catalog_sku_name_normalized": ["iphone", "samsung", "холодильник", "телевизор", "ноутбук", "планшет"],
                "catalog_brand_original": ["Apple", "Samsung", "Test", "Test", "Test", "Test"],
                "catalog_brand": ["apple", "samsung", "test", "test", "test", "test"],
                "catalog_model": ["15 Pro", "S24", "A", "B", "C", "D"],
                "category_id": ["1", "1", "2", "3", "4", "5"],
                "category_name": ["смартфоны", "смартфоны", "холодильники", "телевизоры", "ноутбуки", "планшеты"],
                "category_path": ["электроника > смартфоны"] * 2 + ["техника > холодильники", "электроника > телевизоры", "электроника > ноутбуки", "электроника > планшеты"],
                "available": [True] * 6,
            },
        )

        connection = pipeline.configure_duckdb()
        try:
            pipeline.create_views(connection)
            pipeline.build_query_sku_dataset(connection)
            pipeline.build_no_click_dataset(connection)
            pipeline.build_query_dataset(connection)
            pipeline.build_rejected_dataset(connection)
            pipeline.build_train_valid(connection)
            metrics = pipeline.collect_metrics(
                connection,
                {"catalog_skus": 6},
                {"unique_queries": len(queries), "empty_after_normalization": 0},
            )
            pipeline.build_report(connection, metrics)
        finally:
            connection.close()

        assert metrics["split_overlap"] == 0
        assert metrics["train_queries"] > 0
        assert metrics["valid_queries"] > 0
        assert pipeline.REPORT_PATH.exists()
        print(metrics)


if __name__ == "__main__":
    main()
