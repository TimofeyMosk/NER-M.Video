#!/usr/bin/env python3
"""Verify that the GitHub/runtime package is complete and operational."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "production_manifest.json"

SMOKE_CASES = (
    (
        "хочу купить айфон самсунг темно синий 256 гб",
        {"category": "Смартфоны", "brand": "samsung", "color": "синий"},
    ),
    (
        "телик лджи 55 дюймов 120 гц",
        {"category": "Телевизоры", "brand": "lg", "screen_diagonal": "55 inch", "refresh_rate": "120 hz"},
    ),
    (
        "макбук эпл серый 16 гб",
        {"category": "Ноутбуки", "brand": "apple", "color": "серый"},
    ),
    (
        "чехол для телефона",
        {"category": "Чехлы для телефонов"},
    ),
)


def load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def check_files(manifest: dict[str, object]) -> tuple[list[dict[str, object]], int]:
    missing: list[str] = []
    files: list[dict[str, object]] = []
    total_bytes = 0
    for record in manifest["required_files"]:
        relative = str(record["path"])
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": relative, "bytes": size, "role": record["role"]})
    if missing:
        raise RuntimeError("Missing production files: " + ", ".join(missing))
    return files, total_bytes


def check_annotation(result: dict[str, object], expected: dict[str, str]) -> None:
    model_text = str(result["texts"]["query_model"])
    observed = {str(fact["type"]): str(fact["value"]) for fact in result["facts"]}
    for entity_type, value in expected.items():
        if observed.get(entity_type) != value:
            raise AssertionError(f"Expected {entity_type}={value!r}, got {observed.get(entity_type)!r}")
    for fact in result["facts"]:
        start, end = int(fact["start"]), int(fact["end"])
        if model_text[start:end] != fact["surface"]:
            raise AssertionError(f"Invalid span for {fact}")
    if len(result["tokens"]) != len(result["bio_tags"]) or len(result["tokens"]) != len(result["bio_supervision_mask"]):
        raise AssertionError("Token/BIO/mask lengths differ")


def main() -> int:
    manifest = load_manifest()
    files, total_bytes = check_files(manifest)

    from unified_query_annotator import annotate_query

    started = time.perf_counter()
    results = []
    for query, expected in SMOKE_CASES:
        result = annotate_query(query)
        check_annotation(result, expected)
        results.append(result)
    cold_and_smoke_seconds = time.perf_counter() - started

    benchmark_queries = [query for query, _ in SMOKE_CASES] * 100
    started = time.perf_counter()
    for query in benchmark_queries:
        annotate_query(query)
    benchmark_seconds = time.perf_counter() - started

    report = {
        "status": "PASSED",
        "runtime_version": manifest["runtime_version"],
        "required_files": len(files),
        "runtime_package_bytes": total_bytes,
        "runtime_package_megabytes": round(total_bytes / 1024 / 1024, 3),
        "smoke_cases": len(results),
        "cold_load_plus_smoke_seconds": round(cold_and_smoke_seconds, 4),
        "benchmark_queries": len(benchmark_queries),
        "benchmark_qps": round(len(benchmark_queries) / benchmark_seconds, 1),
        "benchmark_ms_per_query": round(1000 * benchmark_seconds / len(benchmark_queries), 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
