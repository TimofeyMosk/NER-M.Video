import unittest

from search_dictionaries.measurement_parser import MeasurementParser, prepare_model_text


class MeasurementParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = MeasurementParser()

    def candidate(self, query: str, unit: str, category: str = ""):
        result = self.parser.parse(query, category)
        matches = [item for item in result.candidates if item.canonical_unit == unit]
        self.assertTrue(matches, (query, result))
        return matches[0]

    def test_preserves_model_punctuation(self) -> None:
        self.assertEqual(prepare_model_text('  Samsung\tQE55-Q80C 55"  '), 'Samsung QE55-Q80C 55"')

    def test_screen_diagonal_quote(self) -> None:
        item = self.candidate('телевизор Samsung 55"', "inch", "телевизоры")
        self.assertEqual(item.surface, '55"')
        self.assertEqual(item.numbers, ("55",))
        self.assertEqual(item.entity_type, "screen_diagonal")
        self.assertTrue(item.bio_eligible)

    def test_casefold_expansion_does_not_shift_offsets(self) -> None:
        query = 'İ телевизор 55" Straße'
        item = self.candidate(query, "inch")
        self.assertEqual(query[item.start:item.end], item.surface)
        self.assertEqual(item.surface, '55"')

    def test_battery_capacity(self) -> None:
        item = self.candidate("powerbank 5000 мАч", "mah")
        self.assertEqual(item.entity_type, "battery_capacity")
        self.assertEqual(item.numbers, ("5000",))

    def test_unit_does_not_take_number_on_the_right(self) -> None:
        result = self.parser.parse("powerbank 5K Mah 20Вт")
        battery = [item for item in result.candidates if item.canonical_unit == "mah"]
        power = [item for item in result.candidates if item.canonical_unit == "w"]
        self.assertTrue(battery)
        self.assertFalse(battery[0].has_number)
        self.assertFalse(battery[0].bio_eligible)
        self.assertEqual(power[0].numbers, ("20",))

    def test_decimal_and_dimension_specific_number_bounds(self) -> None:
        power = self.candidate("нагреватель 1,5 кВт", "kw")
        self.assertEqual(power.surface, "1,5 кВт")
        self.assertEqual(power.numbers, ("1.5",))
        refresh = self.candidate("монитор 1920x1080/100 Гц", "hz")
        self.assertEqual(refresh.surface, "100 Гц")
        self.assertEqual(refresh.numbers, ("100",))
        punctuation = self.candidate("утюг RI-C284, 2200 Вт", "w")
        self.assertEqual(punctuation.surface, "2200 Вт")
        self.assertEqual(punctuation.numbers, ("2200",))

    def test_refresh_rate_needs_screen_context(self) -> None:
        display = self.candidate("монитор 144 Гц", "hz")
        self.assertEqual(display.entity_type, "refresh_rate")
        generic = self.candidate("генератор 50 Гц", "hz")
        self.assertIsNone(generic.entity_type)
        self.assertFalse(generic.bio_eligible)

    def test_memory_requires_context(self) -> None:
        ambiguous = self.candidate("ноутбук 16 ГБ", "gb")
        self.assertIsNone(ambiguous.entity_type)
        self.assertEqual(set(ambiguous.suggested_entity_types), {"memory_ram", "memory_rom"})
        ram = self.candidate("оперативная память 16 ГБ", "gb")
        self.assertEqual(ram.entity_type, "memory_ram")
        rom = self.candidate("встроенная память 256 ГБ", "gb")
        self.assertEqual(rom.entity_type, "memory_rom")
        mixed = self.parser.parse("RAM 16 ГБ, SSD 512 ГБ")
        typed = [(item.numbers, item.entity_type) for item in mixed.candidates if item.canonical_unit == "gb"]
        self.assertEqual(typed, [(('16',), 'memory_ram'), (('512',), 'memory_rom')])

    def test_model_codes_and_brand_are_rejected(self) -> None:
        for query in ("Samsung S24", "Samsung A55", "iPhone 5S", "ноутбук HP"):
            result = self.parser.parse(query)
            self.assertFalse(result.candidates, query)
            self.assertTrue(result.rejected, query)

    def test_positive_only_bio(self) -> None:
        query = "монитор 144 Гц samsung"
        result = self.parser.parse(query)
        tokens, tags, mask = self.parser.word_bio(query, result.candidates)
        self.assertEqual([row["text"] for row in tokens], ["монитор", "144", "Гц", "samsung"])
        self.assertEqual(tags, ["O", "B-refresh_rate", "I-refresh_rate", "O"])
        self.assertEqual(mask, [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
