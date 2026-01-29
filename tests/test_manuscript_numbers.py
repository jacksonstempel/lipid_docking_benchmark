import unittest


class TestManuscriptNumbers(unittest.TestCase):
    def test_manuscript_matches_analysis_outputs(self) -> None:
        # This is a reproducibility guardrail: if you regenerate analysis outputs
        # and update the manuscript, this should continue to pass.
        from scripts.verify_manuscript_numbers import main

        self.assertEqual(main(), 0)

