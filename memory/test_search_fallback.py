import unittest

from memory.search_fallback import SearchLexicon, TokenExpansionSearchStrategy


class TestTokenExpansionSearchStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TokenExpansionSearchStrategy(
            lexicon=SearchLexicon(
                stopwords=frozenset({"what", "do", "you", "like"}),
                synonyms={
                    "beverage": ("drink", "coffee", "espresso", "cappuccino"),
                    "coffee": ("beverage", "drink", "espresso", "cappuccino"),
                    "espresso": ("coffee", "beverage", "drink", "cappuccino"),
                    "cappuccino": ("coffee", "beverage", "drink", "espresso"),
                },
            )
        )

    def test_semantic_like_overlap_scores_related_text(self):
        score = self.strategy.score(
            "What beverages do you like?",
            "I enjoy drinking coffee in the morning",
        )
        self.assertGreater(score, 0)

    def test_rank_rows_filters_irrelevant_content(self):
        rows = [
            ("user", "I enjoy drinking coffee in the morning"),
            ("user", "Programming is my hobby"),
            ("user", "I like espresso and cappuccino"),
        ]
        ranked = self.strategy.rank_rows(
            "What beverages do you like?",
            rows,
            lambda row: row[1],
        )

        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(item.score > 0 for item in ranked))


if __name__ == "__main__":
    unittest.main()
