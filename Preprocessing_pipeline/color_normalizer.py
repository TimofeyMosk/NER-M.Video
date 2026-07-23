#!/usr/bin/env python3
"""Catalog-backed color normalization and positive-only BIO suggestions.

Character offsets always refer to the minimally cleaned ``query_model`` text.
The optional canonical text is a separate feature because replacements can
change string length.  The implementation uses token tries, never regex.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from search_dictionaries.color_palette import COLOR_NORMALIZATION_VERSION, PALETTE_30


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIASES = ROOT / "search_dictionaries" / "output" / "color_aliases.csv"
DEFAULT_BRANDS = ROOT / "search_dictionaries" / "output" / "brands.json"
AMBIGUOUS_ENGLISH = frozenset({"black", "white", "orange", "red", "gold", "silver", "mint", "cream"})


@dataclass(frozen=True)
class WordToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class AliasRecord:
    color_id: str
    canonical: str
    normalized: str
    source: str
    catalog_count: int
    mapping_reason: str


@dataclass(frozen=True)
class ColorCandidate:
    surface: str
    start: int
    end: int
    token_start: int
    token_end: int
    color_id: str
    canonical: str
    matched_alias: str
    confidence: float
    source: str
    bio_eligible: bool
    reason: str
    normalizer_version: str = COLOR_NORMALIZATION_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ColorResult:
    candidates: tuple[ColorCandidate, ...]
    rejected: tuple[ColorCandidate, ...]
    canonical_text: str


def normalize_token(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("ё", "е").replace("Ё", "Е").casefold()


def tokenize_words(text: str) -> list[WordToken]:
    value = str(text)
    tokens: list[WordToken] = []
    start: int | None = None
    for index, char in enumerate(value):
        if char.isalnum():
            if start is None:
                start = index
        elif start is not None:
            tokens.append(WordToken(normalize_token(value[start:index]), start, index))
            start = None
    if start is not None:
        tokens.append(WordToken(normalize_token(value[start:]), start, len(value)))
    return tokens


def load_brand_patterns(path: Path, minimum_count: int = 20) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    records = json.loads(path.read_text(encoding="utf-8"))
    result: set[tuple[str, ...]] = set()
    for record in records:
        if int(record.get("catalog_count", 0)) < minimum_count:
            continue
        for surface in {str(record.get("canonical", "")), *(str(x) for x in record.get("aliases", []))}:
            pattern = tuple(token.text for token in tokenize_words(surface))
            if pattern:
                result.add(pattern)
    return result


def build_pattern_trie(patterns: set[tuple[str, ...]]) -> dict[str, object]:
    trie: dict[str, object] = {}
    for pattern in patterns:
        node = trie
        for part in pattern:
            node = node.setdefault(part, {})  # type: ignore[assignment]
        node["__end__"] = True
    return trie


def pattern_ranges(tokens: list[WordToken], trie: dict[str, object]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for start in range(len(tokens)):
        node = trie
        index = start
        while index < len(tokens) and tokens[index].text in node:
            node = node[tokens[index].text]  # type: ignore[index,assignment]
            index += 1
            if node.get("__end__"):  # type: ignore[union-attr]
                ranges.append((start, index))
    return ranges


class ColorNormalizer:
    def __init__(self, aliases_path: Path = DEFAULT_ALIASES, brands_path: Path = DEFAULT_BRANDS) -> None:
        if not aliases_path.exists():
            raise FileNotFoundError(
                f"Color aliases not found: {aliases_path}. Run search_dictionaries/build_color_dictionary.py"
            )
        self.aliases_path = aliases_path
        self.palette = {color_id: canonical for color_id, canonical, _ in PALETTE_30}
        self.trie: dict[str, object] = {}
        self.alias_count = 0
        with aliases_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("active", "true").lower() != "true":
                    continue
                pattern = tuple(token.text for token in tokenize_words(row["normalized"]))
                if not pattern:
                    continue
                record = AliasRecord(
                    color_id=row["color_id"], canonical=row["canonical"], normalized=row["normalized"],
                    source=row["source"], catalog_count=int(row["catalog_count"] or 0),
                    mapping_reason=row["mapping_reason"],
                )
                node = self.trie
                for part in pattern:
                    node = node.setdefault(part, {})  # type: ignore[assignment]
                node.setdefault("__records__", []).append(record)  # type: ignore[union-attr]
                self.alias_count += 1
        self.brand_patterns = load_brand_patterns(brands_path)
        self.brand_trie = build_pattern_trie(self.brand_patterns)

    def _matches(self, tokens: list[WordToken]) -> list[tuple[int, int, AliasRecord]]:
        matches: list[tuple[int, int, AliasRecord]] = []
        for start in range(len(tokens)):
            node = self.trie
            index = start
            while index < len(tokens) and tokens[index].text in node:
                node = node[tokens[index].text]  # type: ignore[index,assignment]
                index += 1
                for record in node.get("__records__", ()):  # type: ignore[union-attr]
                    matches.append((start, index, record))
        return matches

    def parse(self, text: str) -> ColorResult:
        value = str(text)
        tokens = tokenize_words(value)
        raw = self._matches(tokens)
        if not raw:
            return ColorResult((), (), value)
        # Brand protection is only needed when a color alias was actually found.
        # This avoids a second trie pass for the overwhelming majority of queries.
        brand_ranges = pattern_ranges(tokens, self.brand_trie)
        by_span: dict[tuple[int, int], list[AliasRecord]] = {}
        for start, end, record in raw:
            by_span.setdefault((start, end), []).append(record)

        proposed: list[tuple[int, int, AliasRecord, bool, str]] = []
        rejected: list[ColorCandidate] = []
        for (start, end), records in by_span.items():
            color_ids = {record.color_id for record in records}
            best = max(records, key=lambda record: (
                record.catalog_count, record.source.startswith("manual"), record.color_id
            ))
            within_brand = any(start >= left and end <= right for left, right in brand_ranges)
            ambiguous = len(color_ids) != 1
            if ambiguous:
                proposed.append((start, end, best, False, "ambiguous_alias"))
            elif within_brand:
                proposed.append((start, end, best, False, "inside_brand"))
            else:
                proposed.append((start, end, best, True, "catalog_or_manual_alias"))

        # Longest-match first, then catalog frequency. Accepted spans cannot overlap.
        proposed.sort(key=lambda item: (-(item[1] - item[0]), -item[2].catalog_count, item[0]))
        accepted_rows: list[tuple[int, int, AliasRecord, str]] = []
        occupied: set[int] = set()
        for start, end, record, eligible, reason in proposed:
            if any(index in occupied for index in range(start, end)):
                continue
            candidate = self._candidate(value, tokens, start, end, record, eligible, reason)
            if eligible:
                accepted_rows.append((start, end, record, reason))
                occupied.update(range(start, end))
            else:
                rejected.append(candidate)

        accepted = tuple(
            sorted(
                (self._candidate(value, tokens, start, end, record, True, reason)
                 for start, end, record, reason in accepted_rows),
                key=lambda item: item.start,
            )
        )
        rejected.sort(key=lambda item: item.start)
        return ColorResult(accepted, tuple(rejected), self.canonicalize_text(value, accepted))

    @staticmethod
    def _candidate(
        text: str, tokens: list[WordToken], start: int, end: int, record: AliasRecord,
        eligible: bool, reason: str,
    ) -> ColorCandidate:
        char_start, char_end = tokens[start].start, tokens[end - 1].end
        is_ambiguous_english = record.normalized in AMBIGUOUS_ENGLISH
        confidence = 0.96 if record.catalog_count else 0.91
        if is_ambiguous_english:
            confidence -= 0.08
        if not eligible:
            confidence = 0.0
        return ColorCandidate(
            surface=text[char_start:char_end], start=char_start, end=char_end,
            token_start=start, token_end=end, color_id=record.color_id,
            canonical=record.canonical, matched_alias=record.normalized,
            confidence=round(confidence, 3), source=record.source,
            bio_eligible=eligible, reason=reason,
        )

    @staticmethod
    def canonicalize_text(text: str, candidates: tuple[ColorCandidate, ...]) -> str:
        output = text
        for item in sorted(candidates, key=lambda value: value.start, reverse=True):
            output = output[:item.start] + item.canonical + output[item.end:]
        return output

    @staticmethod
    def word_bio(text: str, candidates: tuple[ColorCandidate, ...]) -> tuple[list[dict[str, object]], list[str], list[bool]]:
        tokens = tokenize_words(text)
        tags = ["O"] * len(tokens)
        mask = [False] * len(tokens)
        for entity in candidates:
            first = True
            for index, token in enumerate(tokens):
                if token.start < entity.end and token.end > entity.start:
                    tags[index] = ("B-" if first else "I-") + "color"
                    mask[index] = True
                    first = False
        token_rows = [{"text": text[token.start:token.end], "start": token.start, "end": token.end} for token in tokens]
        return token_rows, tags, mask

    def metadata(self) -> dict[str, object]:
        return {
            "version": COLOR_NORMALIZATION_VERSION,
            "palette_size": len(self.palette),
            "alias_rows_loaded": self.alias_count,
            "brand_patterns_loaded": len(self.brand_patterns),
            "regular_expressions": False,
            "span_text": "query_model",
            "canonical_text_offsets": False,
            "supervision_policy": "positive_only_partial_bio",
        }
