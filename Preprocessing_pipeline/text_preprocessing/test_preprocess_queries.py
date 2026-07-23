from __future__ import annotations

import unittest

from text_preprocessing.preprocess_queries import (
    SafeLemmatizer,
    normalize_spaces,
    preprocess,
    replace_hyphens_and_quotes,
)


class PreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lemmatizer = SafeLemmatizer({"самсунг"})

    def test_spaces(self) -> None:
        self.assertEqual(normalize_spaces("  телевизор\t  samsung\n"), "телевизор samsung")

    def test_unicode_separators_without_regex(self) -> None:
        self.assertEqual(replace_hyphens_and_quotes("a—b «c»"), "a b  c ")

    def test_safe_pipeline(self) -> None:
        result = preprocess("  ТЕЛЕВИЗОРЫ—Samsung  ", self.lemmatizer)
        self.assertEqual(result.lemma, "телевизор samsung")

    def test_yo_and_hyphen(self) -> None:
        result = preprocess("Ёлки-палки", self.lemmatizer)
        self.assertEqual(result.lemma, "елка палка")

    def test_model_and_latin_are_preserved(self) -> None:
        result = preprocess("iPhone-15 Pro 43P7K", self.lemmatizer)
        self.assertEqual(result.lemma, "iphone 15 pro 43p7k")

    def test_brand_is_protected(self) -> None:
        result = preprocess("Самсунг телевизоры", self.lemmatizer)
        self.assertEqual(result.lemma, "самсунг телевизор")

    def test_participle_keeps_adjectival_form(self) -> None:
        result = preprocess("встраиваемая техника", self.lemmatizer)
        self.assertEqual(result.lemma, "встраиваемый техника")

    def test_ambiguous_product_words_are_preserved(self) -> None:
        result = preprocess("мини духовой шкаф 30 мин", self.lemmatizer)
        self.assertEqual(result.lemma, "мини духовой шкаф 30 мин")


if __name__ == "__main__":
    unittest.main()
