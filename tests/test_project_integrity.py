from __future__ import annotations

import ast
import csv
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetIntegrityTests(unittest.TestCase):
    def read_csv(self, relative_path: str, delimiter: str = ","):
        path = ROOT / relative_path
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            rows = list(reader)
        return reader.fieldnames, rows

    def test_bank_marketing_schema_and_labels(self):
        fields, rows = self.read_csv("data/bank_marketing.csv", delimiter=";")
        self.assertEqual(len(rows), 41_188)
        self.assertEqual(len(fields or []), 21)
        self.assertEqual(fields[-1], "y")
        self.assertEqual(
            Counter(row["y"] for row in rows),
            {"no": 36_548, "yes": 4_640},
        )
        self.assertFalse(
            any(value == "" for row in rows for value in row.values())
        )

    def test_persian_news_schema_and_labels_if_local_copy_is_present(self):
        path = ROOT / "data/persian_news.csv"

        if not path.exists():
            self.skipTest(
                "Persian News dataset is intentionally not redistributed."
            )

        fields, rows = self.read_csv("data/persian_news.csv")
        self.assertEqual(fields, ["Text", "Topic"])
        self.assertEqual(len(rows), 61_843)
        self.assertEqual(
            Counter(row["Topic"] for row in rows),
            {"Economy": 20_282, "Sport": 21_000, "Tech": 20_561},
        )
        self.assertFalse(
            any(value == "" for row in rows for value in row.values())
        )


class NotebookIntegrityTests(unittest.TestCase):
    NOTEBOOKS = (
        "notebooks/bank_marketing_classification.ipynb",
        "notebooks/persian_news_naive_bayes.ipynb",
    )

    def test_notebooks_are_clean_and_python_cells_parse(self):
        for relative_path in self.NOTEBOOKS:
            with self.subTest(notebook=relative_path):
                path = ROOT / relative_path
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)

                all_source = "\n".join(
                    "".join(cell.get("source", [])) for cell in notebook["cells"]
                )
                self.assertNotIn("/Users/", all_source)
                self.assertNotIn("pip install", all_source)

                for index, cell in enumerate(notebook["cells"]):
                    if cell.get("cell_type") != "code":
                        continue
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])
                    try:
                        ast.parse("".join(cell.get("source", [])))
                    except SyntaxError as error:
                        self.fail(f"{relative_path} cell {index}: {error}")


class PublicDataPolicyTests(unittest.TestCase):
    def test_data_readme_documents_missing_persian_dataset(self):
        path = ROOT / "data/README.md"
        self.assertTrue(path.is_file())

        text = path.read_text(encoding="utf-8")
        self.assertIn("persian_news.csv", text)
        self.assertIn("not redistributed", text.lower())


if __name__ == "__main__":
    unittest.main()
