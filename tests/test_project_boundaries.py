import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def requirement_entries(requirements: Path) -> list[str]:
    return [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "--"))
    ]


class ProjectBoundariesTest(unittest.TestCase):
    def test_shell_entrypoints_use_linux_line_endings(self) -> None:
        offenders = [
            str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("*.sh") if b"\r\n" in path.read_bytes()
        ]
        self.assertEqual(offenders, [])

    def test_runtime_patches_use_stable_line_endings(self) -> None:
        offenders = [
            str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("*.patch") if b"\r\n" in path.read_bytes()
        ]
        self.assertEqual(offenders, [])

    def test_sft_rl_and_serving_share_one_uv_project(self) -> None:
        self.assertFalse((REPO_ROOT / "training" / "pyproject.toml").exists())
        self.assertFalse((REPO_ROOT / "training" / "uv.lock").exists())

    def test_each_project_owns_only_its_runtime_dependencies(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)

        def dependency_names(project: dict) -> set[str]:
            return {
                dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
                for dependency in project["project"]["dependencies"]
            }

        dependencies = dependency_names(project)
        shared = {"dill", "einops", "imageio", "opencv-python", "tensorboard", "timm"}

        self.assertTrue(shared <= dependencies)
        self.assertNotIn("pre-commit", dependencies)
        self.assertIn("torch==2.8.0", project["project"]["dependencies"])
        self.assertIn("transformers==4.57.1", project["project"]["dependencies"])
        self.assertIn("accelerate==1.12.0", project["project"]["dependencies"])
        self.assertIn("huggingface-hub==0.36.2", project["project"]["dependencies"])
        self.assertIn("safetensors==0.7.0", project["project"]["dependencies"])
        self.assertIn("tokenizers==0.22.1", project["project"]["dependencies"])
        self.assertEqual(project["tool"]["uv"]["sources"]["torch"]["index"], "pytorch-cu128")
        self.assertNotIn("workspace", project["tool"]["uv"])

    def test_retired_product_surfaces_stay_absent(self) -> None:
        for relative in (
            "apps",
            "evaluation",
            "examples/t2i",
            "examples/editing",
            "examples/interleave",
            "examples/vqa",
            "src/sensenova_u1/prompt_enhance",
            "src/sensenova_u1_5",
        ):
            self.assertFalse((REPO_ROOT / relative).exists(), relative)

    def test_only_production_serving_submodules_are_declared(self) -> None:
        modules = (REPO_ROOT / ".gitmodules").read_text()
        self.assertEqual(modules.count("[submodule "), 2)
        self.assertIn("serving/third_party/LightLLM", modules)
        self.assertIn("serving/third_party/LightX2V", modules)

    def test_pip_requirements_mirror_direct_pyproject_dependencies(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)

        projects = ((REPO_ROOT / "requirements.txt", project["project"]["dependencies"], "pytorch-cu128"),)

        for requirements, dependencies, cuda_index in projects:
            self.assertTrue(requirements.is_file())
            contents = requirements.read_text()
            self.assertIn("generated from pyproject.toml", contents)
            self.assertNotIn("--hash", contents)
            self.assertNotIn("# via", contents)
            self.assertEqual(requirement_entries(requirements), dependencies)

            if cuda_index is None:
                self.assertNotIn("--extra-index-url", contents)
            else:
                index_url = next(
                    index["url"] for index in project["tool"]["uv"]["index"] if index["name"] == cuda_index
                )
                self.assertIn(f"--extra-index-url {index_url}", contents)

    def test_rl_serving_lock_contains_every_direct_runtime_dependency(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)
        runtime = set(requirement_entries(REPO_ROOT / "docker" / "rl-engine" / "requirements.lock"))
        self.assertTrue(set(project["project"]["dependencies"]) <= runtime)
        self.assertTrue(
            {
                "dill==0.4.1",
                "httpcore==1.0.9",
                "httpx==0.28.1",
                "imageio==2.37.4",
                "timm==1.0.28",
            }
            <= runtime
        )


if __name__ == "__main__":
    unittest.main()
