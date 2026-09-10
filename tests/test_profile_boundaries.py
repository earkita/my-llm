from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = (
    ROOT
    / "skills"
    / "clean-r9700-repo"
    / "scripts"
    / "audit-profile-boundaries.py"
)
SPEC = importlib.util.spec_from_file_location("profile_boundary_auditor", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def profile(*, development: bool, experimental: bool = False) -> dict:
    runtime = {
        "name": "runtime",
        "status": "diagnostic-only" if development else "production-ready",
        "recipe": "recipe",
    }
    if experimental:
        runtime["experimental_modes"] = {"candidate": {}}
    return {
        "name": "profile",
        "status": "development" if development else "production-ready",
        "model": {"name": "model"},
        "runtime": runtime,
    }


class ProfileBoundaryAuditorTests(unittest.TestCase):
    def _write(self, root: Path, area: str, name: str, value: dict) -> None:
        path = root / "profiles" / area / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_allows_multiple_single_runtime_production_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, "production", "one", profile(development=False))
            self._write(root, "production", "two", profile(development=False))

            result = AUDITOR.audit(root)

        self.assertTrue(result["ok"], result["errors"])

    def test_rejects_an_empty_production_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = AUDITOR.audit(Path(temporary))

        self.assertFalse(result["ok"])
        self.assertIn(
            "profiles/production must contain at least one profile",
            result["errors"],
        )

    def test_rejects_experimental_modes_in_production(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root,
                "production",
                "bad",
                profile(development=False, experimental=True),
            )

            result = AUDITOR.audit(root)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any("experimental_modes" in error for error in result["errors"])
        )

    def test_allows_experimental_modes_in_development(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, "production", "good", profile(development=False))
            self._write(
                root,
                "dev",
                "candidate",
                profile(development=True, experimental=True),
            )

            result = AUDITOR.audit(root)

        self.assertTrue(result["ok"], result["errors"])


if __name__ == "__main__":
    unittest.main()
