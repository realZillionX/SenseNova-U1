from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = (ROOT / "README.md", ROOT / "README_CN.md", *sorted((ROOT / "docs").glob("*.md")))


class DocumentationTest(unittest.TestCase):
    def test_local_links_and_images_exist(self) -> None:
        missing: list[str] = []
        for document in MARKDOWN:
            text = document.read_text(encoding="utf-8")
            references = re.findall(r"\]\(([^)]+)\)", text)
            references += re.findall(r'<img[^>]+src="([^"]+)"', text)
            for reference in references:
                target = reference.split("#", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "#")):
                    continue
                path = (document.parent / target).resolve()
                if not path.exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {reference}")
        self.assertEqual(missing, [])

    def test_product_docs_do_not_reference_retired_repositories(self) -> None:
        forbidden = (
            "github.com/OpenSenseNova/SenseNova-U1",
            "github.com/realZillionX/SenseNova-U1.git",
            "mostar-u1-runtime",
        )
        offenders = []
        for document in MARKDOWN:
            text = document.read_text(encoding="utf-8")
            for value in forbidden:
                if value in text:
                    offenders.append(f"{document.relative_to(ROOT)}: {value}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
