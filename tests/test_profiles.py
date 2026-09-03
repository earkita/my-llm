from __future__ import annotations

import json
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r9700 import api, launcher, proxy
from r9700.backends import build_command
from r9700.backends.vllm import environment as vllm_environment
from r9700.config import (
    ConfigurationError,
    ROOT,
    activate_runtime_mode,
    load_profile,
    load_runtime,
)
from r9700.install import _make_venv_entrypoints_relocatable
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
    def test_venv_entrypoints_do_not_embed_recipe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / "recipe" / "venv"
            bin_directory = venv / "bin"
            bin_directory.mkdir(parents=True)
            entrypoint = bin_directory / "tool"
            entrypoint.write_text(
                "#!/some/other/checkout/.runtime/recipes/demo/venv/bin/python\n"
                "print('ok')\n"
            )

            _make_venv_entrypoints_relocatable(venv)

            content = entrypoint.read_text()
            self.assertTrue(content.startswith("#!/bin/sh\n'''exec'"))
            self.assertIn("$(realpath -- \"$0\")", content)
            self.assertNotIn("/some/other/checkout", content)

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
            "vllm_deepseekv4flash_v0.28",
            "vllm_glm53flash_v0.28",
            "vllm_qwen38flash_pr53896",
        }
        self.assertEqual(set(recipe_names()), expected)
        for recipe in expected:
            with self.subTest(recipe=recipe):
                self.assertRegex(
                    recipe,
                    r"^(?:vllm|llamacpp)_[a-z0-9]+_(?:v\d+\.\d+|pr\d+)$",
                )
        for recipe in expected:
            with self.subTest(recipe=recipe):
                verify_assets(recipe_name=recipe)

        for recipe in expected:
            manifest_path = ROOT / "manifest" / f"{recipe}.json"
            runtime_manifest = json.loads(manifest_path.read_text())
            for source in runtime_manifest["sources"].values():
                for patch_record in source["patches"]:
                    patch_path = ROOT / patch_record["path"]
                    self.assertEqual(patch_path.parent.name, recipe)

    def test_glm_uses_an_isolated_repo_local_vllm_recipe(self) -> None:
        runtime = load_profile("glm53-flash")["runtime"]
        manifest = json.loads(
            (ROOT / "manifest/vllm_glm53flash_v0.28.json").read_text()
        )
        self.assertEqual(runtime["recipe"], "vllm_glm53flash_v0.28")
        self.assertEqual(
            manifest["environment"]["venv"],
            ".runtime/recipes/vllm_glm53flash_v0.28/venv",
        )
        self.assertEqual(
            manifest["sources"]["vllm"]["repository"],
            "https://github.com/vllm-project/vllm.git",
        )
        self.assertEqual(
            manifest["sources"]["vllm"]["commit"],
            "c7e6e36fa93a5b8cb95b74fa96e4abdf2f0be51d",
        )
        self.assertEqual(runtime["environment"]["VLLM_USE_V2_MODEL_RUNNER"], "1")

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

    def test_qwen_claude_stack_disables_unstable_long_context_thinking(self) -> None:
        settings = load_profile("qwen38-flash")["stack"]["claude_settings"]
        environment = settings["env"]
        self.assertEqual(
            environment["ANTHROPIC_MODEL"],
            "qwen3.8-flash-next-fast",
        )
        self.assertEqual(environment["CLAUDE_CODE_DISABLE_THINKING"], "1")
        self.assertEqual(environment["MAX_THINKING_TOKENS"], "0")

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
            "glm53-flash": ("vllm", "--quantization", "quark"),
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

    def test_glm_dflash_is_identity_bound_and_explicitly_opt_in(self) -> None:
        profile = load_profile("glm53-flash")
        artifact = profile["model"]["auxiliary_artifacts"][0]
        self.assertEqual(artifact["repository"], "incoai/GLM-5.3-Flash-DFlash2")
        self.assertEqual(artifact["revision"][:7], "bf582e4")
        self.assertNotIn("speculative_config", profile["runtime"])

        runtime = activate_runtime_mode(
            profile["model"], profile["runtime"], "dflash2"
        )
        self.assertEqual(
            profile["runtime"]["experimental_modes"]["dflash2"]["status"],
            "experimental-validated",
        )
        speculative = runtime["speculative_config"]
        self.assertEqual(runtime["active_experimental_mode"], "dflash2")
        self.assertEqual(speculative["model_artifact"], "dflash2-drafter")
        self.assertEqual(speculative["method"], "dflash")
        # The checkpoint block size is eight: one anchor plus seven drafts.
        self.assertEqual(speculative["num_speculative_tokens"], 7)
        self.assertEqual(speculative["draft_tensor_parallel_size"], 8)
        self.assertEqual(speculative["attention_backend"], "TRITON_ATTN")
        self.assertTrue(
            {"0010", "0012", "0017", "0018"}.issubset(
                profile["runtime"]["required_patches"]
            )
        )

    def test_glm_embeds_k1_diagnostic_runtime_modes(self) -> None:
        dflash = load_runtime("glm53-flash", "dflash2-k1")
        self.assertEqual(dflash["active_experimental_mode"], "dflash2-k1")
        self.assertEqual(
            dflash["speculative_config"]["num_speculative_tokens"], 1
        )
        self.assertEqual(dflash["speculative_config"]["method"], "dflash")

        extractor = load_runtime("glm53-flash", "extract-hidden-states-k1")
        speculative = extractor["speculative_config"]
        self.assertEqual(speculative["method"], "extract_hidden_states")
        self.assertEqual(speculative["draft_sample_method"], "greedy")
        self.assertEqual(
            speculative["draft_model_config"]["hf_config"][
                "eagle_aux_hidden_state_layer_ids"
            ],
            [6, 15, 25, 34, 43],
        )

    def test_vllm_speculative_model_resolves_from_identity_bound_artifact(self) -> None:
        profile = load_profile("glm53-flash")
        runtime = activate_runtime_mode(
            profile["model"], profile["runtime"], "dflash2"
        )
        command = build_command(
            profile["model"], runtime, Path("/models/glm"), "127.0.0.1", 8000
        )
        value = command[command.index("--speculative-config") + 1]
        speculative_config = json.loads(value)
        self.assertNotIn("model_artifact", speculative_config)
        self.assertEqual(
            speculative_config["model"],
            str(Path(profile["model"]["auxiliary_artifacts"][0]["path"]).parent),
        )

    def test_glm_is_mrv2_without_prefix_cache(self) -> None:
        profile = load_profile("glm53-flash")
        runtime = profile["runtime"]
        self.assertFalse(runtime["cache"]["prefix_cache"])
        self.assertEqual(runtime["cache"]["cpu_offload_gb"], 0)
        self.assertEqual(runtime["parallel"]["tensor"], 8)
        self.assertEqual(runtime["limits"]["max_model_len"], 32768)
        self.assertEqual(runtime["environment"]["VLLM_USE_V2_MODEL_RUNNER"], "1")
        self.assertEqual(runtime["moe_backend"], "emulation")
        self.assertEqual(runtime["linear_backend"], "emulation")

    def test_glm_model_download_includes_its_chat_template(self) -> None:
        model = load_profile("glm53-flash")["model"]
        self.assertIn("chat_template.jinja", model["required_files"])
        self.assertIn("chat_template.jinja", model["allow_patterns"])

    def test_glm_api_gate_keeps_reasoning_parser_enabled(self) -> None:
        model = load_profile("glm53-flash")["model"]
        body = api._literal_chat_body(model)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertGreaterEqual(body["max_tokens"], 256)

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

    def test_launcher_lists_every_production_profile(self) -> None:
        result = subprocess.run(
            [ROOT / "run", "launcher", "list"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for name in PROFILE_NAMES:
            self.assertIn(name, result.stdout)
        self.assertIn("MAX CONTEXT", result.stdout)

    def test_launcher_start_never_replaces_a_running_profile(self) -> None:
        state = {
            "profile": "glm53-flash",
            "url": "http://127.0.0.1:8000",
        }
        with patch("r9700.launcher.managed_state", return_value=state):
            with self.assertRaisesRegex(
                ConfigurationError, "launcher switch qwen38-flash"
            ):
                launcher.start("qwen38-flash")

    def test_launcher_direct_stop_refuses_to_orphan_litellm(self) -> None:
        with (
            patch(
                "r9700.launcher.proxy.managed_state",
                return_value={"pid": 1234},
            ),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(
                ConfigurationError, "stop --with-litellm"
            ):
                launcher.stop()

        run.assert_not_called()

    def test_launcher_dry_run_uses_persistent_lifecycle_script(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start("deepseek-v4-flash", dry_run=True)

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), launcher.START_SCRIPT)
        self.assertEqual(command[1:3], ["--profile", "deepseek-v4-flash"])
        self.assertIn("--dry-run", command)

    def test_launcher_switch_stops_before_starting_another_profile(self) -> None:
        state = {
            "profile": "glm53-flash",
            "url": "http://127.0.0.1:8000",
        }
        with (
            patch(
                "r9700.launcher.managed_state",
                side_effect=[state, ConfigurationError("stopped")],
            ),
            patch(
                "r9700.launcher.proxy.managed_state",
                side_effect=ConfigurationError("stopped"),
            ),
            patch("r9700.launcher.subprocess.run") as run,
            patch("builtins.print"),
        ):
            run.return_value.returncode = 0
            launcher.switch("qwen38-flash")

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(Path(commands[0][0]), launcher.STOP_SCRIPT)
        self.assertEqual(Path(commands[1][0]), launcher.START_SCRIPT)
        self.assertEqual(commands[1][1:3], ["--profile", "qwen38-flash"])

    def test_launcher_litellm_mode_uses_transactional_stack_manager(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start(
                "qwen38-flash", with_litellm=True, dry_run=True
            )

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), launcher.STACK_SCRIPT)
        self.assertEqual(command[1:4], ["start", "--preset", "qwen38-flash"])
        self.assertIn("--proxy-ready-timeout", command)
        self.assertIn("--dry-run", command)

    def test_launcher_enables_dflash_only_with_explicit_flag(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start("glm53-flash", dry_run=True)
            launcher.start(
                "glm53-flash",
                experimental_dflash2=True,
                dry_run=True,
            )

        default_command = run.call_args_list[0].args[0]
        experimental_command = run.call_args_list[1].args[0]
        self.assertNotIn("--runtime-mode", default_command)
        mode_index = experimental_command.index("--runtime-mode")
        self.assertEqual(experimental_command[mode_index + 1], "dflash2")

    def test_launcher_accepts_a_generic_embedded_diagnostic_mode(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start(
                "glm53-flash",
                runtime_mode="extract-hidden-states-k1",
                dry_run=True,
            )

        command = run.call_args.args[0]
        mode_index = command.index("--runtime-mode")
        self.assertEqual(
            command[mode_index + 1], "extract-hidden-states-k1"
        )

    def test_launcher_rejects_dflash_for_another_profile(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError, "available only for glm53-flash"
        ):
            launcher.start(
                "qwen38-flash",
                experimental_dflash2=True,
                dry_run=True,
            )

    def test_launcher_waits_before_adding_litellm_to_active_model(self) -> None:
        state = {
            "profile": "qwen38-flash",
            "url": "http://127.0.0.1:8000",
        }
        with (
            patch("r9700.launcher.managed_state", return_value=state),
            patch("r9700.launcher.service_wait") as wait,
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start("qwen38-flash", with_litellm=True)

        wait.assert_called_once_with(timeout=900)
        self.assertEqual(Path(run.call_args.args[0][0]), launcher.STACK_SCRIPT)

    def test_proxy_test_reads_the_inference_runtime_state(self) -> None:
        proxy_state = {
            "pid": 1234,
            "probe_url": "http://127.0.0.1:4000",
        }
        runtime_state = {"profile": "qwen38-flash"}
        with (
            patch("r9700.proxy._state", return_value=proxy_state),
            patch("r9700.proxy._identity_alive", return_value=True),
            patch("r9700.proxy._config_matches", return_value=True),
            patch(
                "r9700.proxy.runtime_managed_state",
                return_value=runtime_state,
            ) as managed_runtime,
            patch(
                "r9700.proxy.load_profile",
                return_value={
                    "stack": {
                        "litellm_aliases": [
                            "qwen3.8-flash-next-fp8",
                            "qwen3.8-flash-next",
                        ],
                        "claude_settings": {
                            "env": {"ANTHROPIC_MODEL": "qwen3.8-flash-next"},
                        },
                    },
                },
            ),
            patch(
                "r9700.proxy._get",
                return_value={
                    "data": [
                        {"id": "qwen3.8-flash-next-fp8"},
                        {"id": "qwen3.8-flash-next"},
                    ],
                },
            ),
            patch(
                "r9700.proxy._post",
                return_value={"choices": [{"message": {"content": "OK"}}]},
            ),
            patch("r9700.proxy.read_dotenv", return_value={}),
            patch("r9700.proxy._master_key", return_value="test-key"),
            patch("builtins.print"),
        ):
            proxy.test()

        managed_runtime.assert_called_once_with()

    def test_proxy_maps_legacy_state_without_profile_by_runtime(self) -> None:
        profile = load_profile("qwen38-flash")
        self.assertEqual(
            proxy._active_profile_name(
                {
                    "model": profile["model"]["name"],
                    "runtime": profile["runtime"]["name"],
                }
            ),
            "qwen38-flash",
        )

    def test_launcher_stack_switch_stops_both_components_first(self) -> None:
        state = {
            "profile": "glm53-flash",
            "url": "http://127.0.0.1:8000",
        }
        with (
            patch(
                "r9700.launcher.managed_state",
                side_effect=[state, ConfigurationError("stopped")],
            ),
            patch("r9700.launcher.subprocess.run") as run,
            patch("builtins.print"),
        ):
            run.return_value.returncode = 0
            launcher.switch("qwen38-flash", with_litellm=True)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(Path(commands[0][0]), launcher.STACK_SCRIPT)
        self.assertEqual(commands[0][1], "stop")
        self.assertEqual(Path(commands[1][0]), launcher.STACK_SCRIPT)
        self.assertEqual(commands[1][1:4], ["start", "--preset", "qwen38-flash"])

    def test_launcher_validates_stack_switch_before_stopping(self) -> None:
        state = {"profile": "glm53-flash"}
        with (
            patch("r9700.launcher.managed_state", return_value=state),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(
                ConfigurationError, "unavailable with --with-litellm"
            ):
                launcher.switch(
                    "qwen38-flash",
                    host="127.0.0.1",
                    with_litellm=True,
                )

        run.assert_not_called()

    def test_recipe_root_cannot_be_redirected_outside_the_repository(self) -> None:
        relative = (
            ".runtime/recipes/vllm_deepseekv4flash_v0.28/venv/pyvenv.cfg"
        )
        with patch.dict(
            "os.environ",
            {"R9700_RECIPE_ROOT": "/tmp/r9700-shared-recipes"},
            clear=False,
        ):
            self.assertEqual(
                recipe_artifact_path("vllm_deepseekv4flash_v0.28", relative),
                ROOT
                / ".runtime/recipes/vllm_deepseekv4flash_v0.28/venv/pyvenv.cfg",
            )
            with self.assertRaisesRegex(
                ConfigurationError, "recipe artifact path is outside"
            ):
                recipe_artifact_path("vllm_deepseekv4flash_v0.28", "/tmp/python")

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
