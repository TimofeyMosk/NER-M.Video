#!/usr/bin/env python3
"""Unified query preprocessing and positive-only BIO annotation.

Public API:

    result = annotate_query("купить телевизор Samsung черный 55 дюймов")

Run without arguments for interactive manual testing.  The module deliberately
uses token tries and character scanning instead of regular expressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_dictionaries.color_normalizer import ColorCandidate, ColorNormalizer, WordToken, tokenize_words
from search_dictionaries.measurement_parser import MeasurementCandidate, MeasurementParser, prepare_model_text
from text_preprocessing.preprocess_queries import SafeLemmatizer, load_protected_brands, preprocess


SCHEMA_VERSION = "unified_query_annotation_v1"
BRANDS_PATH = ROOT / "search_dictionaries" / "output" / "brands.json"
CATEGORIES_PATH = ROOT / "search_dictionaries" / "output" / "categories.json"
CATEGORY_ALIAS_OVERRIDES_PATH = ROOT / "search_dictionaries" / "output" / "category_alias_overrides.json"
STOPWORDS_PATH = ROOT / "text_preprocessing" / "stopwords.ru.json"
MAX_QUERY_CHARS = 2_000

ENTITY_PRIORITY = {
    "brand": 110,
    "category": 105,
    "screen_diagonal": 100,
    "refresh_rate": 100,
    "battery_capacity": 100,
    "power": 95,
    "memory_ram": 95,
    "memory_rom": 95,
    "color": 80,
    "measurement": 60,
}


@dataclass(frozen=True)
class DictionaryEntry:
    entity_type: str
    entity_id: str
    canonical: str
    catalog_count: int


@dataclass(frozen=True)
class DictionaryMatch:
    entity_type: str
    entity_id: str
    canonical: str
    surface: str
    start: int
    end: int
    confidence: float
    source: str
    ambiguity_count: int = 1
    alternatives: tuple[dict[str, str], ...] = ()

    def to_fact(self) -> dict[str, object]:
        value = asdict(self)
        value["alternatives"] = list(self.alternatives)
        value["value"] = self.canonical
        value["type"] = self.entity_type
        return value


class DictionaryTrieMatcher:
    """Longest-match lookup for catalog brands or explicit categories."""

    def __init__(
        self,
        path: Path,
        entity_type: str,
        lemmatizer: SafeLemmatizer,
        alias_overrides_path: Path | None = None,
    ) -> None:
        self.path = path
        self.entity_type = entity_type
        self.lemmatizer = lemmatizer
        self.alias_overrides_path = alias_overrides_path
        self.trie: dict[str, object] = {}
        self.patterns = 0
        self.records = 0
        self.override_patterns = 0
        self._load()

    @staticmethod
    def _usable_pattern(pattern: tuple[str, ...]) -> bool:
        joined = "".join(pattern)
        if not joined or not any(char.isalpha() for char in joined):
            return False
        if len(pattern) == 1 and len(joined) < 2:
            return False
        return True

    def _pattern(self, surface: str) -> tuple[str, ...]:
        stages = preprocess(surface, self.lemmatizer)
        return tuple(token.text for token in tokenize_words(stages.lemma))

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        raw_records = json.loads(self.path.read_text(encoding="utf-8"))
        overrides: dict[tuple[str, ...], dict[str, object]] = {}
        if self.alias_overrides_path is not None and self.alias_overrides_path.exists():
            override_payload = json.loads(self.alias_overrides_path.read_text(encoding="utf-8"))
            for surface, metadata in override_payload.get("aliases", {}).items():
                pattern = self._pattern(str(surface))
                if pattern:
                    overrides[pattern] = metadata
        patterns: dict[tuple[str, ...], dict[tuple[str, str], DictionaryEntry]] = defaultdict(dict)
        for record in raw_records:
            if self.entity_type == "brand":
                entity_id = str(record["canonical"])
                canonical = str(record["canonical"])
            else:
                entity_id = str(record["category_id"])
                canonical = str(record["name_original"])
            entry = DictionaryEntry(
                entity_type=self.entity_type,
                entity_id=entity_id,
                canonical=canonical,
                catalog_count=int(record.get("catalog_count", 0)),
            )
            surfaces = {
                canonical,
                str(record.get("normalized", "")),
                str(record.get("lemma", "")),
                *(str(value) for value in record.get("aliases", [])),
            }
            for surface in surfaces:
                pattern = self._pattern(surface)
                if self._usable_pattern(pattern):
                    patterns[pattern][(entry.entity_id, entry.canonical)] = entry
            self.records += 1

        for pattern, entries_by_id in patterns.items():
            entries = tuple(entries_by_id.values())
            source = f"catalog_{self.entity_type}_trie"
            override = overrides.get(pattern)
            if override is not None:
                target_ids = {str(value) for value in override.get("target_ids", [])}
                preferred = tuple(entry for entry in entries if entry.entity_id in target_ids)
                if preferred:
                    entries = preferred
                    sources = "+".join(str(value) for value in override.get("sources", []))
                    source = f"curated_category_alias:{sources or 'configured'}"
                    self.override_patterns += 1
            node = self.trie
            for part in pattern:
                node = node.setdefault(part, {})  # type: ignore[assignment]
            node["__entries__"] = entries
            node["__source__"] = source
            self.patterns += 1

    def _token_keys(self, text: str) -> tuple[list[WordToken], list[str], list[str]]:
        tokens = tokenize_words(text)
        normalized = [token.text for token in tokens]
        lemmas = [self.lemmatizer.lemmatize_token(token.text) for token in tokens]
        return tokens, normalized, lemmas

    def _find_stream(
        self,
        keys: list[str],
    ) -> list[tuple[int, int, tuple[DictionaryEntry, ...], str]]:
        matches: list[tuple[int, int, tuple[DictionaryEntry, ...], str]] = []
        for start in range(len(keys)):
            node = self.trie
            index = start
            while index < len(keys) and keys[index] in node:
                node = node[keys[index]]  # type: ignore[index,assignment]
                index += 1
                entries = node.get("__entries__")  # type: ignore[union-attr]
                if entries:
                    source = str(node.get("__source__", f"catalog_{self.entity_type}_trie"))  # type: ignore[union-attr]
                    matches.append((start, index, entries, source))
        return matches

    def match(self, text: str) -> tuple[list[DictionaryMatch], list[DictionaryMatch]]:
        tokens, normalized, lemmas = self._token_keys(text)
        raw = self._find_stream(normalized) + self._find_stream(lemmas)
        grouped: dict[tuple[int, int], dict[tuple[str, str], DictionaryEntry]] = defaultdict(dict)
        sources_by_span: dict[tuple[int, int], set[str]] = defaultdict(set)
        for start, end, entries, source in raw:
            for entry in entries:
                grouped[(start, end)][(entry.entity_id, entry.canonical)] = entry
            sources_by_span[(start, end)].add(source)

        accepted: list[DictionaryMatch] = []
        rejected: list[DictionaryMatch] = []
        for (token_start, token_end), by_id in grouped.items():
            entries = sorted(by_id.values(), key=lambda item: (-item.catalog_count, item.canonical))
            char_start, char_end = tokens[token_start].start, tokens[token_end - 1].end
            surface = text[char_start:char_end]
            best = entries[0]
            alternatives = tuple(
                {"id": entry.entity_id, "canonical": entry.canonical} for entry in entries
            )
            semantic_values = {entry.canonical.casefold().replace("ё", "е") for entry in entries}
            ambiguous = len(semantic_values) > 1
            sources = sources_by_span[(token_start, token_end)]
            curated_source = next((source for source in sorted(sources) if source.startswith("curated_")), None)
            match = DictionaryMatch(
                entity_type=self.entity_type,
                entity_id=best.entity_id,
                canonical=best.canonical,
                surface=surface,
                start=char_start,
                end=char_end,
                confidence=0.0 if ambiguous else (
                    0.98 if self.entity_type == "brand" else (
                        0.96 if curated_source else (0.92 if len(entries) > 1 else 0.94)
                    )
                ),
                source=curated_source or f"catalog_{self.entity_type}_trie",
                ambiguity_count=len(entries),
                alternatives=alternatives if len(entries) > 1 else (),
            )
            (rejected if ambiguous else accepted).append(match)

        # Keep the longest/highest-frequency interpretation for internal overlaps.
        accepted.sort(key=lambda item: (-(item.end - item.start), -item.confidence, item.start))
        selected: list[DictionaryMatch] = []
        occupied: set[int] = set()
        for item in accepted:
            positions = set(range(item.start, item.end))
            if positions & occupied:
                continue
            occupied.update(positions)
            selected.append(item)
        selected.sort(key=lambda item: item.start)
        rejected.sort(key=lambda item: item.start)
        return selected, rejected

    def metadata(self) -> dict[str, object]:
        return {
            "records": self.records,
            "patterns": self.patterns,
            "override_patterns": self.override_patterns,
            "path": str(self.path),
            "alias_overrides_path": str(self.alias_overrides_path) if self.alias_overrides_path else None,
        }


class StopwordProcessor:
    def __init__(self, path: Path = STOPWORDS_PATH) -> None:
        self.path = path
        data = json.loads(path.read_text(encoding="utf-8"))
        self.version = int(data["version"])
        self.protected = set(str(value) for value in data["protected_not_stop"])
        self.groups: dict[str, set[str]] = {
            name: set(str(value) for value in data[name])
            for name in ("prepositions", "transaction_words", "search_filler", "channel_words")
        }
        self.conditional = {
            str(name): set(str(value) for value in values)
            for name, values in data["conditional_intent_words"].items()
        }

    def process(self, lemmatized_text: str) -> tuple[str, list[dict[str, str]], dict[str, object]]:
        tokens = tokenize_words(lemmatized_text)
        kept: list[str] = []
        removed: list[dict[str, str]] = []
        intent_matches: dict[str, list[str]] = defaultdict(list)
        for token in tokens:
            value = token.text
            if value in self.protected:
                kept.append(value)
                continue
            conditional_group = next((name for name, words in self.conditional.items() if value in words), None)
            if conditional_group is not None:
                intent_matches[conditional_group].append(value)
                removed.append({"token": value, "group": f"intent:{conditional_group}"})
                continue
            group = next((name for name, words in self.groups.items() if value in words), None)
            if group is not None:
                removed.append({"token": value, "group": group})
            else:
                kept.append(value)
        intents = {
            f"{name}_intent": bool(intent_matches.get(name))
            for name in self.conditional
        }
        intents["matched_tokens"] = dict(intent_matches)
        return " ".join(kept), removed, intents


class UnifiedQueryAnnotator:
    """Orchestrates normalization, dictionary facts and unified partial BIO."""

    def __init__(self) -> None:
        self.lemmatizer = SafeLemmatizer(load_protected_brands(BRANDS_PATH))
        self.stopwords = StopwordProcessor()
        self.brands = DictionaryTrieMatcher(BRANDS_PATH, "brand", self.lemmatizer)
        self.categories = DictionaryTrieMatcher(
            CATEGORIES_PATH,
            "category",
            self.lemmatizer,
            CATEGORY_ALIAS_OVERRIDES_PATH,
        )
        self.measurements = MeasurementParser()
        self.colors = ColorNormalizer()

    @staticmethod
    def _fact_key(item: dict[str, object]) -> tuple[object, ...]:
        return (item["type"], item["start"], item["end"], item.get("value"))

    @staticmethod
    def _measurement_fact(candidate: MeasurementCandidate) -> dict[str, object]:
        entity_type = candidate.entity_type or "measurement"
        numbers = list(candidate.numbers)
        canonical_value = " ".join([*numbers, candidate.canonical_unit]).strip()
        return {
            "type": entity_type,
            "value": canonical_value,
            "surface": candidate.surface,
            "start": candidate.start,
            "end": candidate.end,
            "confidence": round(float(candidate.confidence if candidate.entity_type else 0.88), 3),
            "source": candidate.source,
            "numbers": numbers,
            "canonical_unit": candidate.canonical_unit,
            "preferred_unit": candidate.preferred_unit,
            "dimension": candidate.dimension,
            "suggested_entity_types": list(candidate.suggested_entity_types),
            "reason": candidate.reason,
        }

    @staticmethod
    def _color_fact(candidate: ColorCandidate) -> dict[str, object]:
        return {
            "type": "color",
            "value": candidate.canonical,
            "color_id": candidate.color_id,
            "surface": candidate.surface,
            "start": candidate.start,
            "end": candidate.end,
            "confidence": candidate.confidence,
            "source": candidate.source,
            "matched_alias": candidate.matched_alias,
        }

    @staticmethod
    def _align_to_tokens(text: str, facts: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], list[bool]]:
        tokens = tokenize_words(text)

        def bio_priority(item: dict[str, object]) -> int:
            source = str(item.get("source", ""))
            if item.get("type") == "category" and "product_family" in source:
                return 115
            return ENTITY_PRIORITY.get(str(item["type"]), 0)

        ranked = sorted(
            facts,
            key=lambda item: (
                -bio_priority(item),
                -(int(item["end"]) - int(item["start"])),
                -float(item["confidence"]),
                int(item["start"]),
            ),
        )
        selected: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        occupied: set[int] = set()
        for fact in ranked:
            indexes = [
                index for index, token in enumerate(tokens)
                if token.start < int(fact["end"]) and token.end > int(fact["start"])
            ]
            if not indexes:
                rejected.append({**fact, "rejection_reason": "no_word_token_alignment"})
                continue
            if any(index in occupied for index in indexes):
                rejected.append({**fact, "rejection_reason": "overlap_with_higher_priority_fact"})
                continue
            entity = {**fact, "token_start": indexes[0], "token_end": indexes[-1] + 1}
            selected.append(entity)
            occupied.update(indexes)
        selected.sort(key=lambda item: (int(item["token_start"]), int(item["token_end"])))

        tags = ["O"] * len(tokens)
        mask = [False] * len(tokens)
        for entity in selected:
            for offset, index in enumerate(range(int(entity["token_start"]), int(entity["token_end"]))):
                tags[index] = f"{'B' if offset == 0 else 'I'}-{entity['type']}"
                mask[index] = True
        token_rows = [
            {"text": text[token.start:token.end], "normalized": token.text, "start": token.start, "end": token.end}
            for token in tokens
        ]
        return selected, rejected, tags, mask

    @staticmethod
    def _group_facts(facts: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for fact in facts:
            grouped[str(fact["type"])].append(fact)
        return dict(grouped)

    def annotate(self, text: str, category_hint: str = "") -> dict[str, object]:
        original = str(text)
        if len(original) > MAX_QUERY_CHARS:
            raise ValueError(f"Query is too long: {len(original)} > {MAX_QUERY_CHARS}")
        stages = preprocess(original, self.lemmatizer)
        model_text = prepare_model_text(original)
        search_text, removed_stopwords, intents = self.stopwords.process(stages.lemma)
        model_input_color_result = self.colors.parse(search_text)

        brand_matches, brand_rejected = self.brands.match(model_text)
        category_matches, category_rejected = self.categories.match(model_text)
        inferred_hint = category_hint or (category_matches[0].canonical if category_matches else "")
        measurement_result = self.measurements.parse(model_text, inferred_hint)
        color_result = self.colors.parse(model_text)

        measurement_facts = [
            self._measurement_fact(item)
            for item in measurement_result.candidates
            if item.has_number
        ]
        accepted_brand_facts: list[dict[str, object]] = []
        unit_overlap_rejected: list[dict[str, object]] = []
        for item in brand_matches:
            fact = item.to_fact()
            inside_measurement = any(
                int(fact["start"]) >= int(measurement["start"])
                and int(fact["end"]) <= int(measurement["end"])
                for measurement in measurement_facts
            )
            if inside_measurement:
                unit_overlap_rejected.append({
                    **fact,
                    "rejection_reason": "brand_surface_inside_number_plus_unit",
                })
            else:
                accepted_brand_facts.append(fact)

        facts: list[dict[str, object]] = []
        facts.extend(accepted_brand_facts)
        facts.extend(item.to_fact() for item in category_matches)
        facts.extend(measurement_facts)
        facts.extend(self._color_fact(item) for item in color_result.candidates)

        unique_facts: dict[tuple[object, ...], dict[str, object]] = {}
        for fact in facts:
            key = self._fact_key(fact)
            previous = unique_facts.get(key)
            if previous is None or float(fact["confidence"]) > float(previous["confidence"]):
                unique_facts[key] = fact
        facts = sorted(unique_facts.values(), key=lambda item: (int(item["start"]), int(item["end"]), str(item["type"])))
        entities, overlap_rejected, tags, mask = self._align_to_tokens(model_text, facts)
        bio_keys = {self._fact_key(entity) for entity in entities}
        facts = [
            {**fact, "selected_for_bio": self._fact_key(fact) in bio_keys}
            for fact in facts
        ]
        tokens = tokenize_words(model_text)

        rejected_candidates: list[dict[str, object]] = []
        rejected_candidates.extend({**item.to_fact(), "rejection_reason": "ambiguous_dictionary_surface"} for item in brand_rejected)
        rejected_candidates.extend({**item.to_fact(), "rejection_reason": "ambiguous_dictionary_surface"} for item in category_rejected)
        rejected_candidates.extend({**item.to_dict(), "type": "measurement", "rejection_reason": item.reason} for item in measurement_result.rejected)
        rejected_candidates.extend(
            {**item.to_dict(), "type": item.entity_type or "measurement", "rejection_reason": "unit_without_number"}
            for item in measurement_result.candidates if not item.has_number
        )
        rejected_candidates.extend({**item.to_dict(), "type": "color", "rejection_reason": item.reason} for item in color_result.rejected)
        rejected_candidates.extend(unit_overlap_rejected)
        rejected_candidates.extend(overlap_rejected)

        warnings: list[str] = []
        if not original.strip():
            warnings.append("empty_query")
        if category_rejected:
            warnings.append("ambiguous_explicit_category")
        if any(fact["type"] == "measurement" and fact.get("suggested_entity_types") for fact in facts):
            warnings.append("generic_measurement_requires_model_context")

        return {
            "schema_version": SCHEMA_VERSION,
            "input": original,
            "texts": {
                "query_model": model_text,
                "query_normalized": stages.spaces,
                "query_preprocessed": stages.lemma,
                "query_search_text": search_text,
                "query_color_canonical": color_result.canonical_text,
                "query_model_input": model_input_color_result.canonical_text,
            },
            "intents": intents,
            "removed_stopwords": removed_stopwords,
            "facts": facts,
            "facts_by_type": self._group_facts(facts),
            "entities": entities,
            "tokens": [
                {"text": model_text[token.start:token.end], "normalized": token.text, "start": token.start, "end": token.end}
                for token in tokens
            ],
            "bio_tags": tags,
            "bio_supervision_mask": mask,
            "rejected_candidates": rejected_candidates,
            "warnings": warnings,
            "meta": {
                "positive_only_partial_bio": True,
                "category_hint": inferred_hint,
                "regular_expressions": False,
                "brand_dictionary": self.brands.metadata(),
                "category_dictionary": self.categories.metadata(),
                "measurement_parser": self.measurements.metadata()["version"],
                "color_normalizer": self.colors.metadata()["version"],
                "stopwords_version": self.stopwords.version,
            },
        }


@lru_cache(maxsize=1)
def get_default_annotator() -> UnifiedQueryAnnotator:
    """Load dictionaries once per process."""
    return UnifiedQueryAnnotator()


def annotate_query(text: str, category_hint: str = "") -> dict[str, object]:
    """Preprocess one query and return facts plus unified partial BIO."""
    return get_default_annotator().annotate(text, category_hint)


def print_human(result: dict[str, object]) -> None:
    texts = result["texts"]
    print("\n" + "=" * 88)
    print(f"INPUT       : {result['input']}")
    print(f"MODEL       : {texts['query_model']}")
    print(f"NORMALIZED  : {texts['query_normalized']}")
    print(f"PREPROCESSED: {texts['query_preprocessed']}")
    print(f"SEARCH TEXT : {texts['query_search_text']}")
    print(f"COLOR TEXT  : {texts['query_color_canonical']}")
    print(f"MODEL INPUT : {texts['query_model_input']}")

    print("\nFACTS")
    facts = result["facts"]
    if not facts:
        print("  — факты не найдены")
    for fact in facts:
        details = ""
        if fact["type"] == "measurement":
            details = f" [{fact.get('dimension')}; {fact.get('canonical_unit')}]"
        print(
            f"  {str(fact['type']):20} {fact['surface']!r} -> {fact['value']!r}"
            f"  span={fact['start']}:{fact['end']}  conf={float(fact['confidence']):.3f}{details}"
        )

    print("\nUNIFIED PARTIAL BIO")
    print(f"  {'TOKEN':22} {'TAG':28} {'MASK':5} SPAN")
    print("  " + "-" * 78)
    for token, tag, mask in zip(result["tokens"], result["bio_tags"], result["bio_supervision_mask"]):
        print(f"  {token['text'][:22]:22} {tag:28} {str(mask):5} {token['start']}:{token['end']}")

    removed = result["removed_stopwords"]
    if removed:
        print("\nREMOVED STOPWORDS")
        print("  " + ", ".join(f"{item['token']} [{item['group']}]" for item in removed))
    active_intents = [name for name, value in result["intents"].items() if name != "matched_tokens" and value]
    if active_intents:
        print("INTENTS      : " + ", ".join(active_intents))
    if result["warnings"]:
        print("WARNINGS     : " + ", ".join(result["warnings"]))
    print(f"Rejected candidates: {len(result['rejected_candidates'])}")


def append_jsonl(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified preprocessing and partial BIO annotation")
    parser.add_argument("query", nargs="*", help="Query text. Without it, interactive mode starts.")
    parser.add_argument("--category-hint", default="", help="Optional category context for measurements")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the human table")
    parser.add_argument("--output", type=Path, help="Append every result to one UTF-8 JSONL file")
    return parser.parse_args()


def handle_query(query: str, args: argparse.Namespace) -> None:
    result = annotate_query(query, args.category_hint)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    if args.output:
        append_jsonl(args.output, result)


def main() -> int:
    args = parse_args()
    if args.query:
        handle_query(" ".join(args.query), args)
        return 0

    print("Единый annotator готов. Введите запрос; пустая строка завершает работу.")
    print("Пример: купить телевизор Samsung темно-синий 55 дюймов 120 Гц")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        try:
            handle_query(query, args)
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
