#!/usr/bin/env python3
"""Production measurement candidate parser for query preprocessing and BIO NER.

The parser deliberately uses character scanning and a token trie instead of
regular expressions. It returns character spans first; word-level BIO tags are
derived from those spans and are explicitly marked as positive-only partial
supervision.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from search_dictionaries.units import UNIT_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANDS = ROOT / "search_dictionaries" / "output" / "brands.json"
PARSER_VERSION = "measurement_trie_v1"
BRAND_COLLISION_MIN_COUNT = 50
SYMBOL_TOKENS = frozenset({'"', "'", "″", "′", "%", "°", "Ω", "ω", "µ", ".", ",", "/", "*", "·"})
AMBIGUOUS_WITHOUT_NUMBER = frozenset({'"', "'", "in"})
NUMERIC_CONNECTOR_TOKENS = frozenset({".", ",", "/", "*", "x", "х", "×", "+", "-"})

SCREEN_CONTEXT = (
    "телевиз", "монитор", "ноутбук", "планшет", "смартфон", "телефон",
    "экран", "диспле", "диагонал", "проектор", "smart tv", " tv",
    "monitor", "laptop", "tablet", "phone", "screen", "display",
)
REFRESH_CONTEXT = SCREEN_CONTEXT + ("частот", "обновлен", "refresh",)
RAM_CONTEXT = ("оператив", "озу", " ram", "ram ", "ddr",)
ROM_CONTEXT = ("встроен", "постоянн", " rom", "rom ", "пзу", "ssd", "накопител", "storage")


@dataclass(frozen=True)
class Token:
    text: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class AliasEntry:
    canonical: str
    dimension: str
    preferred: str
    alias: str
    pattern: tuple[str, ...]
    requires_number: bool
    always_ambiguous: bool


@dataclass(frozen=True)
class MeasurementCandidate:
    surface: str
    start: int
    end: int
    unit_surface: str
    unit_start: int
    unit_end: int
    numbers: tuple[str, ...]
    number_surface: str
    canonical_unit: str
    preferred_unit: str
    dimension: str
    has_number: bool
    entity_type: str | None
    suggested_entity_types: tuple[str, ...]
    bio_eligible: bool
    confidence: float
    source: str
    reason: str
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["numbers"] = list(self.numbers)
        value["suggested_entity_types"] = list(self.suggested_entity_types)
        return value


@dataclass(frozen=True)
class ParseResult:
    candidates: tuple[MeasurementCandidate, ...]
    rejected: tuple[MeasurementCandidate, ...]
    has_any_number: bool


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).replace("ё", "е").replace("Ё", "Е").casefold()


def prepare_model_text(text: str) -> str:
    """Minimal model input cleanup that preserves meaningful punctuation."""
    value = unicodedata.normalize("NFKC", str(text))
    output: list[str] = []
    pending_space = False
    for char in value:
        if unicodedata.category(char) in {"Cc", "Cf"} or char.isspace():
            pending_space = bool(output)
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
        output.append(char)
    return "".join(output).strip()


def token_kind(char: str) -> str | None:
    if char.isdigit():
        return "number"
    if char.isalpha():
        return "word"
    if char in SYMBOL_TOKENS:
        return "symbol"
    return None


def tokenize(text: str) -> list[Token]:
    value = prepare_model_text(text)
    tokens: list[Token] = []
    index = 0
    while index < len(value):
        kind = token_kind(value[index])
        if kind is None:
            index += 1
            continue
        if kind == "symbol":
            token_text = value[index]
            if token_text == "″":
                token_text = '"'
            elif token_text == "′":
                token_text = "'"
            tokens.append(Token(token_text.casefold(), kind, index, index + 1))
            index += 1
            continue
        start = index
        index += 1
        while index < len(value) and token_kind(value[index]) == kind:
            index += 1
        tokens.append(Token(normalize_text(value[start:index]), kind, start, index))
    return tokens


def canonical_decimal(value: str) -> str | None:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def extract_numbers(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text)).replace("−", "-").replace("—", "-")
    result: list[str] = []
    buffer = ""
    has_decimal = False

    def flush() -> None:
        nonlocal buffer, has_decimal
        if any(char.isdigit() for char in buffer):
            normalized = canonical_decimal(buffer.replace(",", "."))
            if normalized is not None:
                result.append(normalized)
        buffer = ""
        has_decimal = False

    for index, char in enumerate(value):
        if char.isdigit():
            buffer += char
        elif char in {".", ","} and buffer and not has_decimal:
            buffer += "."
            has_decimal = True
        elif char in {"+", "-"} and not buffer and index + 1 < len(value) and value[index + 1].isdigit():
            buffer = char
        else:
            flush()
    flush()
    return result


def alias_is_single_letter(pattern: tuple[str, ...]) -> bool:
    joined = "".join(pattern)
    letters = "".join(char for char in joined if char.isalpha())
    return bool(letters) and len(letters) == 1 and all(char.isalpha() for char in joined)


def load_brand_patterns(path: Path, minimum_count: int) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    records = json.loads(path.read_text(encoding="utf-8"))
    patterns: set[tuple[str, ...]] = set()
    for record in records:
        if int(record.get("catalog_count", 0)) < minimum_count:
            continue
        surfaces = {str(record["canonical"]), *(str(value) for value in record.get("aliases", []))}
        for surface in surfaces:
            pattern = tuple(token.text for token in tokenize(surface))
            if pattern:
                patterns.add(pattern)
    return patterns


def context_contains(text: str, markers: tuple[str, ...]) -> bool:
    normalized = " " + normalize_text(text) + " "
    return any(marker in normalized for marker in markers)


def infer_entity_type(
    canonical_unit: str,
    has_number: bool,
    query_text: str,
    category_hint: str,
    local_context: str = "",
) -> tuple[str | None, tuple[str, ...], float, str]:
    if not has_number:
        return None, (), 0.55, "unit_without_adjacent_number"
    context = f"{query_text} {category_hint}"
    if canonical_unit in {"mah", "ah"}:
        return "battery_capacity", ("battery_capacity",), 0.98, "electric_charge_unit"
    if canonical_unit == "inch" and context_contains(context, SCREEN_CONTEXT):
        return "screen_diagonal", ("screen_diagonal",), 0.96, "inch_with_screen_context"
    if canonical_unit == "cm" and context_contains(context, ("диагонал", "screen diagonal")):
        return "screen_diagonal", ("screen_diagonal",), 0.93, "centimeter_with_diagonal_context"
    if canonical_unit == "hz" and context_contains(context, REFRESH_CONTEXT):
        return "refresh_rate", ("refresh_rate",), 0.94, "hertz_with_display_context"
    if canonical_unit in {"mw", "w", "kw", "hp"}:
        return "power", ("power",), 0.93, "power_unit"
    if canonical_unit in {"mb", "gb", "tb"}:
        memory_context = f"{local_context} {category_hint}"
        if context_contains(memory_context, RAM_CONTEXT):
            return "memory_ram", ("memory_ram",), 0.94, "data_size_with_ram_context"
        if context_contains(memory_context, ROM_CONTEXT):
            return "memory_rom", ("memory_rom",), 0.94, "data_size_with_rom_context"
        return None, ("memory_ram", "memory_rom"), 0.70, "ambiguous_memory_type"
    return None, (), 0.65, "measurement_without_project_bio_type"


def separators_only(text: str) -> bool:
    return not any(char.isalpha() or char.isdigit() for char in text)


class MeasurementParser:
    def __init__(
        self,
        brand_path: Path = DEFAULT_BRANDS,
        brand_collision_min_count: int = BRAND_COLLISION_MIN_COUNT,
    ) -> None:
        self.brand_path = Path(brand_path)
        self.brand_collision_min_count = int(brand_collision_min_count)
        self.trie: dict[str, object] = {}
        self.unit_pattern_collisions: dict[str, list[str]] = {}
        self.brand_collisions: list[str] = []
        self.usable_alias_patterns = 0
        self._build_trie()

    def _build_trie(self) -> None:
        brand_patterns = load_brand_patterns(self.brand_path, self.brand_collision_min_count)
        brand_collisions: set[str] = set()
        pattern_entries: dict[tuple[str, ...], list[AliasEntry]] = defaultdict(list)
        for definition in UNIT_DEFINITIONS:
            canonical = str(definition["canonical"])
            dimension = str(definition["dimension"])
            preferred = str(definition["preferred"])
            aliases = {canonical, preferred, *(str(value) for value in definition["aliases"])}
            for alias in aliases:
                pattern = tuple(token.text for token in tokenize(alias))
                if not pattern:
                    continue
                brand_collision = pattern in brand_patterns
                if brand_collision:
                    brand_collisions.add(" ".join(pattern))
                pattern_entries[pattern].append(AliasEntry(
                    canonical=canonical,
                    dimension=dimension,
                    preferred=preferred,
                    alias=alias,
                    pattern=pattern,
                    requires_number="".join(pattern) in AMBIGUOUS_WITHOUT_NUMBER,
                    always_ambiguous=alias_is_single_letter(pattern) or brand_collision,
                ))

        for pattern, entries in pattern_entries.items():
            canonical_units = sorted({entry.canonical for entry in entries})
            if len(canonical_units) > 1:
                self.unit_pattern_collisions[" ".join(pattern)] = canonical_units
                continue
            entry = sorted(
                entries,
                key=lambda item: (item.always_ambiguous, item.requires_number, len(item.alias), item.alias),
            )[0]
            node = self.trie
            for part in pattern:
                node = node.setdefault(part, {})  # type: ignore[assignment]
            node["__entry__"] = entry
            self.usable_alias_patterns += 1
        self.brand_collisions = sorted(brand_collisions)

    @staticmethod
    def _has_adjacent_number(tokens: list[Token], start_index: int, end_index: int, text: str) -> bool:
        if start_index > 0 and tokens[start_index - 1].kind == "number":
            if separators_only(text[tokens[start_index - 1].end:tokens[start_index].start]):
                return True
        return False

    @staticmethod
    def _measurement_bounds(
        tokens: list[Token],
        start_index: int,
        end_index: int,
        text: str,
        entry: AliasEntry,
    ) -> tuple[int, int]:
        left = start_index
        right = end_index
        if left > 0 and tokens[left - 1].kind == "number" and separators_only(text[tokens[left - 1].end:tokens[left].start]):
            left -= 1
            allowed_connectors = {".", ","}
            if entry.dimension == "data_size":
                allowed_connectors.add("/")
            if entry.dimension in {"length", "resolution"} and entry.canonical != "inch":
                allowed_connectors.update({"x", "х", "×", "*"})
            while left >= 2:
                connector = tokens[left - 1]
                number = tokens[left - 2]
                if number.kind != "number" or connector.text not in allowed_connectors:
                    break
                if connector.text in {".", ","}:
                    if number.end != connector.start or connector.end != tokens[left].start:
                        break
                elif not separators_only(text[number.end:connector.start]) or not separators_only(text[connector.end:tokens[left].start]):
                    break
                left -= 2
        return tokens[left].start, tokens[right - 1].end

    @staticmethod
    def _local_context(text: str, start: int, end: int) -> str:
        boundaries = frozenset(",;+/|()[]{}")
        left = start
        while left > 0 and text[left - 1] not in boundaries:
            left -= 1
        right = end
        while right < len(text) and text[right] not in boundaries:
            right += 1
        return text[left:right]

    def parse(self, text: str, category_hint: str = "") -> ParseResult:
        model_text = prepare_model_text(text)
        tokens = tokenize(model_text)
        candidates: list[MeasurementCandidate] = []
        rejected: list[MeasurementCandidate] = []
        has_any_number = any(token.kind == "number" for token in tokens)
        index = 0
        while index < len(tokens):
            node: dict[str, object] = self.trie
            cursor = index
            best: tuple[int, AliasEntry] | None = None
            while cursor < len(tokens) and tokens[cursor].text in node:
                node = node[tokens[cursor].text]  # type: ignore[assignment]
                cursor += 1
                entry = node.get("__entry__")
                if isinstance(entry, AliasEntry):
                    best = (cursor, entry)
            if best is None:
                index += 1
                continue
            end_index, entry = best
            numeric = self._has_adjacent_number(tokens, index, end_index, model_text)
            measurement_start, measurement_end = self._measurement_bounds(tokens, index, end_index, model_text, entry)
            surface = model_text[measurement_start:measurement_end]
            numbers = tuple(extract_numbers(surface)) if numeric else ()
            entity_type, suggestions, confidence, reason = infer_entity_type(
                entry.canonical,
                numeric,
                model_text,
                category_hint,
                self._local_context(model_text, measurement_start, measurement_end),
            )
            rejected_by_policy = entry.always_ambiguous or (entry.requires_number and not numeric)
            if rejected_by_policy:
                entity_type = None
                suggestions = ()
                confidence = 0.0
                reason = "ambiguous_unit_alias"
            candidate = MeasurementCandidate(
                surface=surface,
                start=measurement_start,
                end=measurement_end,
                unit_surface=model_text[tokens[index].start:tokens[end_index - 1].end],
                unit_start=tokens[index].start,
                unit_end=tokens[end_index - 1].end,
                numbers=numbers,
                number_surface=model_text[measurement_start:tokens[index].start].strip() if numeric else "",
                canonical_unit=entry.canonical,
                preferred_unit=entry.preferred,
                dimension=entry.dimension,
                has_number=numeric,
                entity_type=entity_type,
                suggested_entity_types=suggestions,
                bio_eligible=bool(entity_type and numeric and not rejected_by_policy),
                confidence=confidence,
                source=PARSER_VERSION,
                reason=reason,
            )
            (rejected if rejected_by_policy else candidates).append(candidate)
            index = end_index
        return ParseResult(tuple(candidates), tuple(rejected), has_any_number)

    def word_bio(
        self,
        text: str,
        candidates: tuple[MeasurementCandidate, ...] | list[MeasurementCandidate],
    ) -> tuple[list[dict[str, object]], list[str], list[bool]]:
        model_text = prepare_model_text(text)
        tokens = [token for token in tokenize(model_text) if token.kind in {"word", "number"}]
        tags = ["O"] * len(tokens)
        mask = [False] * len(tokens)
        eligible = sorted(
            (candidate for candidate in candidates if candidate.bio_eligible),
            key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
        )
        occupied: set[int] = set()
        for candidate in eligible:
            indexes = [
                index for index, token in enumerate(tokens)
                if token.start < candidate.end and token.end > candidate.start
            ]
            if not indexes or any(index in occupied for index in indexes):
                continue
            for offset, index in enumerate(indexes):
                tags[index] = f"{'B' if offset == 0 else 'I'}-{candidate.entity_type}"
                mask[index] = True
                occupied.add(index)
        token_rows = [
            {"text": model_text[token.start:token.end], "start": token.start, "end": token.end}
            for token in tokens
        ]
        return token_rows, tags, mask

    def metadata(self) -> dict[str, object]:
        return {
            "version": PARSER_VERSION,
            "unit_definitions": len(UNIT_DEFINITIONS),
            "usable_alias_patterns": self.usable_alias_patterns,
            "unit_pattern_collisions": self.unit_pattern_collisions,
            "brand_colliding_alias_patterns": self.brand_collisions,
            "regular_expressions": False,
            "bio_policy": "positive-only high-confidence spans; O tokens are unsupervised",
        }
