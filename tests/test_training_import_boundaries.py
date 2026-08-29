from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrainingImportBoundariesTest(unittest.TestCase):
    def test_model_package_does_not_eagerly_create_the_engine_cycle(self) -> None:
        source = ROOT / "training" / "sensenovavl" / "model" / "sensenovavl_moe_chat" / "__init__.py"
        tree = ast.parse(source.read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertNotIn("modeling_neo_vit", imported)
        self.assertNotIn("modeling_sensenovavl_chat_mot", imported)


if __name__ == "__main__":
    unittest.main()
