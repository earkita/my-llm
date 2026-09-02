from __future__ import annotations

import json
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r9700.backends import build_command
from r9700.backends.vllm import environment as vllm_environment
from r9700.config import ConfigurationError, ROOT, load_profile
from r9700.manifest import (
    recipe_artifact_path,
    recipe_names,
    sha256_file,
    verify_assets,
)
from r9700.model_worker import validate as validate_checkpoint
from r9700.models import verify_model
from r9700.service import start


PROFILE_NAMES = (
    "deepseek-v4-flash",
    "glm53-flash",
    "qwen38-flash",
)


class ProductionProfileTests(unittest.TestCase):
    def test_repository_contains_exactly_three_flat_profiles(self) -> None:
        root = ROOT / "profiles" / "production"
        self.assertEqual(
            [path.stem for path in sorted(root.glob("*.json"))],
            list(PROFILE_NAMES),
        )
        self.assertFalse((ROOT / "profiles" / "models").exists())
        self.assertFalse((ROOT / "profiles" / "runtime").exists())
        self.assertFalse((ROOT / "config" / "stack-presets.json").exists())

    def test_profiles_are_self_contained_and_have_no_inheritance(self) -> None:
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                path = ROOT / "profiles" / "production" / f"{name}.json"
                raw = path.read_text()
                self.assertNotIn('"extends"', raw)
                profile = load_profile(name)
                self.assertEqual(profile["status"], "production-ready")
                self.assertIsInstance(profile["model"], dict)
                self.assertIsInstance(profile["runtime"], dict)
                self.assertIsInstance(
                    profile["stack"]["claude_settings"], dict
                )

    def test_profile_loader_rejects_inheritance_at_any_depth(self) -> None:
        path = ROOT / "tests" / "invalid-flat-profile.json"
        source = json.loads(
            (ROOT / "profiles" / "production" / "glm53-flash.json").read_text()
        )
        source["name"] = path.stem
        source["stack"]["claude_settings"]["extends"] = "forbidden.json"
        path.write_text(json.dumps(source))
        try:
            with self.assertRaisesRegex(ConfigurationError, "cannot use extends"):
                load_profile(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_only_required_recipes_and_assets_are_present(self) -> None:
        expected = {
            "llama-cpp-glm53-pr27754",
            "vllm-dspark-v0280",
            "vllm-qwen38-flash-next-v0280-pr53896",
        }
        self.assertEqual(set(recipe_names()), expected)
        for recipe in expected - {"llama-cpp-glm53-pr27754"}:
            with self.subTest(recipe=recipe):
                verify_assets(recipe_name=recipe)

        llama_manifest = json.loads(
            (ROOT / "manifest" / "llama-cpp-glm53-pr27754.json").read_text()
        )
        for patch in llama_manifest["patches"]:
            path = ROOT / patch["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), patch["sha256"])

    def test_provenance_hashes_current_flat_profiles(self) -> None:
        provenance = json.loads((ROOT / "provenance.json").read_text())
        for name, record in provenance["profiles"].items():
            with self.subTest(profile=name):
                path = ROOT / record["target_profile"]
                self.assertEqual(sha256_file(path), record["target_profile_sha256"])

    def test_litellm_config_matches_embedded_aliases(self) -> None:
        expected = {
            alias
            for name in PROFILE_NAMES
            for alias in load_profile(name)["stack"]["litellm_aliases"]
        }
        configured = set(
            re.findall(
                r"^\s*- model_name:\s*(\S+)\s*$",
                (ROOT / "config" / "litellm.yaml").read_text(),
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(configured, expected)

    def test_no_production_profile_enables_cpu_offload(self) -> None:
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                runtime = load_profile(name)["runtime"]
                self.assertEqual(runtime["cache"].get("cpu_offload_gb", 0), 0)
                self.assertNotIn("weight_offload", runtime)

    def test_profile_loader_rejects_cpu_offload(self) -> None:
        path = ROOT / "tests" / "invalid-offload-profile.json"
        source = json.loads(
            (ROOT / "profiles" / "production" / "qwen38-flash.json").read_text()
        )
        source["name"] = path.stem
        source["runtime"]["cache"]["cpu_offload_gb"] = 1
        path.write_text(json.dumps(source))
        try:
            with self.assertRaisesRegex(ConfigurationError, "cannot use CPU offload"):
                load_profile(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_commands_resolve_from_one_profile(self) -> None:
        expectations = {
            "deepseek-v4-flash": ("vllm", "--pipeline-parallel-size", "6"),
            "glm53-flash": ("llama-server", "--spec-type", "draft-dflash"),
            "qwen38-flash": ("vllm", "--tensor-parallel-size", "8"),
        }
        for name, expected in expectations.items():
            with self.subTest(profile=name):
                profile = load_profile(name)
                command = build_command(
                    profile["model"],
                    profile["runtime"],
                    Path("/models") / name,
                    "127.0.0.1",
                    8000,
                )
                rendered = " ".join(command)
                for value in expected:
                    self.assertIn(value, rendered)
                if profile["runtime"].get("backend", "vllm") == "vllm":
                    self.assertEqual(
                        command[1:3], ["-m", "r9700.vllm_entrypoint"]
                    )

    def test_vllm_workers_receive_the_rocm_triton_bootstrap(self) -> None:
        runtime = {
            "recipe": "fixture",
            "transport": {
                "p2p_disable": 1,
                "shm_disable": 0,
                "socket_ifname": "lo",
                "runtime_connect": 1,
                "hsa_legacy_ipc": 0,
            },
            "shutdown": {
                "worker_timeout_seconds": 60,
                "process_grace_seconds": 60,
            },
            "parallel": {},
            "environment": {},
        }
        with (
            patch(
                "r9700.backends.vllm.base_environment",
                return_value={"PATH": "/bin", "PYTHONPATH": "/existing"},
            ),
            patch("r9700.backends.vllm.rocm_root", return_value=Path("/rocm")),
            patch(
                "r9700.backends.vllm.recipe_venv",
                return_value=Path("/recipe/venv"),
            ),
            patch("r9700.backends.vllm.visible_devices", return_value=["0"]),
        ):
            env = vllm_environment(runtime)

        paths = env["PYTHONPATH"].split(":")
        self.assertEqual(
            paths[:2], [str(ROOT / "r9700/vllm_bootstrap"), str(ROOT)]
        )
        self.assertEqual(paths[2], "/existing")
        self.assertTrue((Path(paths[0]) / "sitecustomize.py").is_file())

    def test_service_rejects_cross_profile_composition(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "model and runtime must come from the same production profile",
        ):
            start("deepseek-v4-flash", "qwen38-flash")

    def test_glm_dflash_artifact_is_identity_bound(self) -> None:
        profile = load_profile("glm53-flash")
        artifact = profile["model"]["auxiliary_artifacts"][0]
        runtime_artifact = profile["runtime"]["dflash2_artifact"]
        self.assertEqual(
            profile["runtime"]["llama_cpp"]["draft_model"],
            artifact["path"],
        )
        for key in ("repository", "revision", "filename", "size_bytes", "sha256"):
            self.assertEqual(artifact[key], runtime_artifact[key])

    def test_cli_lists_only_production_profiles(self) -> None:
        result = subprocess.run(
            [ROOT / "run", "profiles", "list", "--json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        records = json.loads(result.stdout)
        self.assertEqual([record["name"] for record in records], list(PROFILE_NAMES))
        self.assertTrue(all(record["tier"] == "production" for record in records))

    def test_shared_recipe_root_preserves_recipe_boundaries(self) -> None:
        shared = Path("/tmp/r9700-shared-recipes")
        relative = (
            ".runtime/recipes/vllm-dspark-v0280/venv/bin/python"
        )
        with patch.dict(
            "os.environ", {"R9700_RECIPE_ROOT": str(shared)}, clear=False
        ):
            self.assertEqual(
                recipe_artifact_path("vllm-dspark-v0280", relative),
                shared / "vllm-dspark-v0280/venv/bin/python",
            )
            with self.assertRaisesRegex(
                ConfigurationError, "recipe artifact path is outside"
            ):
                recipe_artifact_path("vllm-dspark-v0280", "/tmp/python")

    def test_model_verification_rechecks_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            shard = destination / "model-00001.safetensors"
            shard.write_bytes(b"test checkpoint")
            model = {
                "name": "fixture",
                "repository": "example/model",
                "revision": "0" * 40,
                "expected_shards": 1,
                "weight_pattern": "*.safetensors",
                "required_files": [],
                "_sha256": "1" * 64,
            }
            source = {
                "repository": model["repository"],
                "revision": model["revision"],
                "profile_sha256": model["_sha256"],
                "checkpoint": validate_checkpoint(model, destination),
            }
            (destination / ".model-source.json").write_text(
                json.dumps(source)
            )
            with (
                patch("r9700.models.load_model", return_value=model),
                patch(
                    "r9700.models.resolve_model_directory",
                    return_value=destination,
                ),
            ):
                verify_model("fixture")
                shard.unlink()
                with self.assertRaisesRegex(
                    ConfigurationError, "checkpoint validation failed"
                ):
                    verify_model("fixture")

    def test_safetensors_weight_bytes_exclude_container_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            header = json.dumps(
                {
                    "weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
                separators=(",", ":"),
            ).encode()
            shard = destination / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
            model = {
                "expected_shards": 1,
                "weight_pattern": "*.safetensors",
                "required_files": [],
                "checkpoint_weight_bytes": 4,
            }
            self.assertEqual(
                validate_checkpoint(model, destination)["weight_bytes"],
                shard.stat().st_size,
            )
            model["checkpoint_weight_bytes"] = 5
            with self.assertRaisesRegex(RuntimeError, "tensor bytes differ"):
                validate_checkpoint(model, destination)


if __name__ == "__main__":
    unittest.main()
