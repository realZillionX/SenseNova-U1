from __future__ import annotations

import re
import unittest
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))


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

    def test_installation_uses_declared_repository(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        repository = project["urls"]["Repository"]
        installation = (ROOT / "docs/installation.md").read_text()
        self.assertIn(f"git clone {repository}.git", installation)
        self.assertIn(f"cd {repository.rsplit('/', 1)[-1]}", installation)


if __name__ == "__main__":
    unittest.main()
