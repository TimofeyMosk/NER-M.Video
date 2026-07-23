from __future__ import annotations

import unittest

from unified_query_annotator import annotate_query, get_default_annotator


class UnifiedQueryAnnotatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.annotator = get_default_annotator()

    def assert_valid_spans_and_bio(self, result: dict[str, object]) -> None:
        model_text = result["texts"]["query_model"]
        for fact in result["facts"]:
            self.assertEqual(model_text[fact["start"]:fact["end"]], fact["surface"])
        self.assertEqual(len(result["tokens"]), len(result["bio_tags"]))
        self.assertEqual(len(result["tokens"]), len(result["bio_supervision_mask"]))
        for tag, mask in zip(result["bio_tags"], result["bio_supervision_mask"]):
            self.assertEqual(tag != "O", mask)

    def test_full_query(self) -> None:
        result = annotate_query(
            "хочу купить телевизор Samsung темно-синий 55 дюймов 120 Гц со скидкой"
        )
        types = {fact["type"] for fact in result["facts"]}
        self.assertTrue({"category", "brand", "color", "screen_diagonal", "refresh_rate"}.issubset(types))
        self.assertEqual(result["texts"]["query_search_text"], "телевизор samsung темно синий 55 дюйм 120 гц")
        self.assertEqual(result["texts"]["query_model_input"], "телевизор samsung синий 55 дюйм 120 гц")
        self.assertTrue(result["intents"]["promotion_intent"])
        self.assertFalse(result["intents"]["price_intent"])
        self.assert_valid_spans_and_bio(result)

    def test_generic_measurement_and_unit_brand_collision(self) -> None:
        result = annotate_query("купить стол 120x60 см коричневый")
        measurements = [fact for fact in result["facts"] if fact["type"] == "measurement"]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]["canonical_unit"], "cm")
        self.assertFalse(any(fact["type"] == "brand" and fact["surface"].casefold() == "см" for fact in result["facts"]))
        self.assertTrue(any(
            item.get("rejection_reason") == "brand_surface_inside_number_plus_unit"
            for item in result["rejected_candidates"]
        ))
        self.assert_valid_spans_and_bio(result)

    def test_brand_phrase_protects_color(self) -> None:
        result = annotate_query("дрель black decker 500 вт")
        self.assertTrue(any(fact["type"] == "brand" and fact["value"] == "black+decker" for fact in result["facts"]))
        self.assertFalse(any(fact["type"] == "color" for fact in result["facts"]))
        self.assertTrue(any(fact["type"] == "power" for fact in result["facts"]))
        self.assert_valid_spans_and_bio(result)

    def test_protected_stopwords_stay(self) -> None:
        result = annotate_query("пылесос без мешка до 30000")
        search_tokens = result["texts"]["query_search_text"].split()
        self.assertIn("без", search_tokens)
        self.assertIn("до", search_tokens)

    def test_ambiguous_memory_is_generic(self) -> None:
        result = annotate_query("ноутбук HP 16 ГБ серый")
        self.assertTrue(any(fact["type"] == "measurement" for fact in result["facts"]))
        self.assertIn("generic_measurement_requires_model_context", result["warnings"])
        self.assert_valid_spans_and_bio(result)

    def test_empty_query(self) -> None:
        result = annotate_query("   ")
        self.assertEqual(result["facts"], [])
        self.assertIn("empty_query", result["warnings"])

    def test_category_synonyms_share_one_canonical_value(self) -> None:
        for query in ("смартфон", "телефон", "айфон", "iphone", "mobile phone"):
            with self.subTest(query=query):
                result = annotate_query(query)
                categories = [fact for fact in result["facts"] if fact["type"] == "category"]
                self.assertEqual([fact["value"] for fact in categories], ["Смартфоны"])
                self.assert_valid_spans_and_bio(result)

    def test_product_family_aliases_map_to_broad_categories(self) -> None:
        cases = {
            "макбук": "Ноутбуки",
            "airpods": "Наушники",
            "эпл вотч": "Смарт-часы",
            "плейстейшен": "Игровые консоли",
            "xbox": "Игровые консоли",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                result = annotate_query(query)
                category = next(fact for fact in result["facts"] if fact["type"] == "category")
                self.assertEqual(category["value"], expected)
                selected = next(entity for entity in result["entities"] if entity["type"] == "category")
                self.assertEqual(selected["value"], expected)
                self.assert_valid_spans_and_bio(result)

    def test_long_accessory_category_beats_generic_phone_alias(self) -> None:
        result = annotate_query("чехол для телефона")
        categories = [fact for fact in result["facts"] if fact["type"] == "category"]
        self.assertEqual([fact["value"] for fact in categories], ["Чехлы для телефонов"])
        self.assert_valid_spans_and_bio(result)

    def test_expanded_cyrillic_brand_aliases(self) -> None:
        cases = {"лджи": "lg", "асер": "acer", "электролюкс": "electrolux", "джибиэль": "jbl"}
        for query, expected in cases.items():
            with self.subTest(query=query):
                result = annotate_query(query)
                brands = [fact["value"] for fact in result["facts"] if fact["type"] == "brand"]
                self.assertEqual(brands, [expected])
                self.assert_valid_spans_and_bio(result)


if __name__ == "__main__":
    unittest.main()
