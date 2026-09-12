from __future__ import annotations

import json
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r9700 import api, cli, launcher, proxy, worker_pool
from r9700.backends import build_command
from r9700.backends.vllm import environment as vllm_environment
from r9700.config import (
    ConfigurationError,
    ROOT,
    load_profile,
    load_runtime,
)
from r9700.install import (
    _amdsmi_install_requirement,
    _make_venv_entrypoints_relocatable,
    install,
)
from r9700.litellm_tools import (
    enforce_glm53_strict_tools,
    local_anthropic_count_tokens_endpoint,
    normalize_qwen38_reasoning_effort,
)
from r9700.manifest import (
    recipe_artifact_path,
    recipe_names,
    sha256_file,
    verify_assets,
)
from r9700.model_worker import validate as validate_checkpoint
from r9700.models import verify_model
from r9700.service import start


REQUIRED_PROFILE_NAMES = {
    "deepseek-v4-flash",
    "glm53-flash",
    "glm53-flash-uncensored",
    "qwen38-flash",
    "qwen38-flash-uncensored",
    "qwen38-4x27b",
    "qwen-multi",
}
PROFILE_NAMES = tuple(
    sorted(
        path.stem
        for path in (ROOT / "profiles" / "production").glob("*.json")
    )
)
GLM_PROFILE = "glm53-flash"
GLM_V029_EXPERIMENTS = str(
    ROOT / "profiles" / "dev" / "glm53-flash-v029-experiments.json"
)


class ProductionProfileTests(unittest.TestCase):
    @patch("r9700.install._install_vllm")
    def test_install_all_skips_backends_without_a_registered_recipe(
        self, install_vllm
    ) -> None:
        install(dry_run=True)

        install_vllm.assert_called_once_with(
            "vllm_deepseekv4flash_v0.28",
            dry_run=True,
            jobs=None,
            rebuild=False,
        )

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

    def test_amdsmi_can_be_installed_from_the_pinned_rocm_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rocm_home = Path(temporary)
            source = rocm_home / "share" / "amd_smi"
            source.mkdir(parents=True)
            (source / "pyproject.toml").write_text("[build-system]\n")
            self.assertEqual(
                _amdsmi_install_requirement(
                    {"amdsmi_from_rocm_sdk": True}, rocm_home=str(rocm_home)
                ),
                str(source),
            )

    def test_repository_contains_expected_flat_profiles(self) -> None:
        root = ROOT / "profiles" / "production"
        self.assertTrue(
            REQUIRED_PROFILE_NAMES.issubset(
                {path.stem for path in root.glob("*.json")}
            )
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
            (
                ROOT
                / "profiles"
                / "production"
                / "glm53-flash.json"
            ).read_text()
        )
        source["name"] = path.stem
        source["stack"]["claude_settings"]["extends"] = "forbidden.json"
        path.write_text(json.dumps(source))
        try:
            with self.assertRaisesRegex(ConfigurationError, "cannot use extends"):
                load_profile(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_profile_loader_rejects_claude_agent_on_undeclared_alias(self) -> None:
        path = ROOT / "tests" / "invalid-claude-agent-profile.json"
        source = json.loads(
            (ROOT / "profiles" / "production" / "qwen-multi.json").read_text()
        )
        source["name"] = path.stem
        source["stack"]["claude_agents"]["qwen-worker-explorer"][
            "model"
        ] = "undeclared-model"
        path.write_text(json.dumps(source))
        try:
            with self.assertRaisesRegex(
                ConfigurationError, "undeclared LiteLLM alias"
            ):
                load_profile(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_only_required_recipes_and_assets_are_present(self) -> None:
        expected = {
            "vllm_deepseekv4flash_v0.28",
            "vllm_glm53flashrocm10_v0.29",
            "vllm_glm53flashrocm10_v0.30",
            "vllm_glm53flashrocm10_v0.31",
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

    def test_every_constraint_file_has_a_consumer(self) -> None:
        referenced = {proxy.REQUIREMENTS_PATH.resolve()}
        for recipe in recipe_names():
            manifest = json.loads(
                (ROOT / "manifest" / f"{recipe}.json").read_text()
            )
            environment = manifest["environment"]
            referenced.add((ROOT / environment["constraints"]).resolve())
            referenced.update(
                (ROOT / overlay["path"]).resolve()
                for overlay in environment.get("constraint_overlays", [])
            )

        present = {path.resolve() for path in (ROOT / "constraints").glob("*")}
        self.assertEqual(present, referenced)

    def test_glm_uses_an_isolated_repo_local_vllm_recipe(self) -> None:
        runtime = load_profile(GLM_PROFILE)["runtime"]
        manifest = json.loads(
            (ROOT / "manifest/vllm_glm53flashrocm10_v0.29.json").read_text()
        )
        self.assertEqual(runtime["recipe"], "vllm_glm53flashrocm10_v0.29")
        self.assertEqual(
            manifest["environment"]["venv"],
            ".runtime/recipes/vllm_glm53flashrocm10_v0.29/venv",
        )
        self.assertEqual(
            manifest["sources"]["vllm"]["repository"],
            "https://github.com/vllm-project/vllm.git",
        )
        self.assertEqual(
            manifest["sources"]["vllm"]["commit"],
            "7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719",
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

    def test_litellm_glm_aliases_declare_qualified_context(self) -> None:
        config = (ROOT / "config" / "litellm.yaml").read_text()
        block = config.split("- model_name: glm-5.3-flash-high", 1)[1].split(
            "\n  - model_name:", 1
        )[0]
        self.assertIn(
            "model: os.environ/HOSTED_INFERENCE_ANTHROPIC_MODEL", block
        )
        self.assertIn("max_input_tokens: 524288", block)

    def test_litellm_binds_the_shared_glm_alias_to_the_active_profile(self) -> None:
        profile_name, model = proxy._active_anthropic_model(
            {"profile": "glm53-flash-uncensored"}
        )

        self.assertEqual(profile_name, "glm53-flash-uncensored")
        self.assertEqual(
            model, "anthropic/glm-5.3-flash-uncensored-quark-mxfp4"
        )

    def test_litellm_forces_strict_glm_tools_without_rewriting_schemas(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        }
        request = {
            "model": "glm-5.3-flash-high",
            "tools": [
                {"name": "Read", "input_schema": schema},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        }

        updated = enforce_glm53_strict_tools(request)

        self.assertIsNot(updated, request)
        self.assertTrue(updated["tools"][0]["strict"])
        self.assertFalse(
            updated["tools"][0]["input_schema"]["additionalProperties"]
        )
        self.assertNotIn("strict", updated["tools"][1])
        self.assertNotIn("strict", request["tools"][0])

    def test_litellm_does_not_force_strict_tools_for_other_models(self) -> None:
        request = {
            "model": "qwen3.8-flash-next",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "probe", "parameters": {}},
                }
            ],
        }

        self.assertIs(enforce_glm53_strict_tools(request), request)

    def test_litellm_anthropic_token_counting_stays_on_local_backend(self) -> None:
        self.assertEqual(
            local_anthropic_count_tokens_endpoint("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1/messages/count_tokens",
        )

    def test_claude_templates_cover_every_runtime_context_variant(self) -> None:
        variant_names = {
            262144: "256k",
            695040: "aligned-695040",
            786432: "768k-vision",
            1048576: "1m",
        }
        profile_contexts: dict[str, dict[int, dict]] = {}
        profile_sources = [*PROFILE_NAMES, GLM_V029_EXPERIMENTS]
        for profile_source in profile_sources:
            profile = load_profile(profile_source)
            profile_name = profile["name"]
            runtimes = [profile["runtime"]]
            runtimes.extend(
                load_runtime(profile_source, mode_name)
                for mode_name in profile["runtime"].get(
                    "experimental_modes", {}
                )
            )
            for runtime in runtimes:
                context_tokens = runtime["limits"]["max_model_len"]
                profile_contexts.setdefault(profile_name, {})[
                    context_tokens
                ] = profile

        expected_templates: dict[tuple[str, str], tuple[dict, int]] = {}
        for profile_name, contexts in profile_contexts.items():
            if profile_name.startswith("glm53-flash"):
                model_directory = "glm53-flash"
            elif profile_name.startswith("qwen38-flash"):
                model_directory = "qwen38-flash"
            else:
                model_directory = profile_name
            for context_tokens, profile in contexts.items():
                suffix = ""
                if len(contexts) > 1:
                    suffix = f"-{variant_names[context_tokens]}"
                filename = f"{profile_name}{suffix}.settings.local.json"
                expected_templates[(model_directory, filename)] = (
                    profile,
                    context_tokens,
                )

        templates_root = ROOT / "templates" / ".claude"
        actual_templates = {
            (path.parent.name, path.name)
            for path in templates_root.glob("*/*.settings.local.json")
        }
        self.assertEqual(actual_templates, set(expected_templates))
        self.assertFalse((templates_root / "settings.local.json").exists())
        self.assertFalse(list(templates_root.glob("*/*/settings.local.json")))

        model_keys = (
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
        )
        for (model_directory, filename), (
            profile,
            context_tokens,
        ) in expected_templates.items():
            with self.subTest(model=model_directory, template=filename):
                template = json.loads(
                    (templates_root / model_directory / filename).read_text()
                )
                expected = json.loads(
                    json.dumps(profile["stack"]["claude_settings"])
                )
                expected["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(
                    context_tokens
                )
                expected["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
                    min(context_tokens, 1_000_000)
                )
                expected["env"]["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "90"
                self.assertEqual(template, expected)
                aliases = profile["stack"]["litellm_aliases"]
                model_names = {template["env"][key] for key in model_keys}
                if model_directory == "glm53-flash":
                    self.assertEqual(model_names, {"glm-5.3-flash-high"})
                elif len(aliases) == 1:
                    self.assertEqual(model_names, {aliases[0]})

    def test_claude_agent_templates_match_embedded_profiles(self) -> None:
        templates_root = ROOT / "templates" / ".claude"
        expected = {
            profile["name"]: profile["stack"]["claude_agents"]
            for name in PROFILE_NAMES
            if "claude_agents" in (profile := load_profile(name))["stack"]
        }
        actual_paths = {
            path.name.removesuffix(".agents.json"): path
            for path in templates_root.glob("*/*.agents.json")
        }

        self.assertEqual(set(actual_paths), set(expected))
        for profile_name, agents in expected.items():
            with self.subTest(profile=profile_name):
                self.assertEqual(
                    json.loads(actual_paths[profile_name].read_text()), agents
                )

    def test_qwen_claude_stack_splits_thinking_and_fast_roles(self) -> None:
        for name in ("qwen38-flash", "qwen38-flash-uncensored"):
            with self.subTest(profile=name):
                settings = load_profile(name)["stack"]["claude_settings"]
                environment = settings["env"]
                self.assertEqual(
                    environment["ANTHROPIC_MODEL"],
                    "qwen3.8-flash-next-thinking",
                )
                self.assertEqual(
                    environment["ANTHROPIC_DEFAULT_SONNET_MODEL"],
                    "qwen3.8-flash-next-thinking",
                )
                self.assertEqual(
                    environment["ANTHROPIC_DEFAULT_OPUS_MODEL"],
                    "qwen3.8-flash-next-thinking",
                )
                self.assertEqual(
                    environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"],
                    "qwen3.8-flash-next-fast",
                )
                self.assertEqual(
                    environment["ANTHROPIC_SMALL_FAST_MODEL"],
                    "qwen3.8-flash-next-fast",
                )
                self.assertNotIn("CLAUDE_CODE_DISABLE_THINKING", environment)
                self.assertNotIn("MAX_THINKING_TOKENS", environment)
                self.assertEqual(settings["effortLevel"], "high")

    def test_qwen_reasoning_effort_normalizes_without_mutation(self) -> None:
        models = (
            "qwen3.8-flash-next-thinking",
            "hosted_vllm/qwen3.8-flash-next-uncensored-mxfp4-fp8",
            "qwen3.8-27b-workers-thinking",
            "hosted_vllm/qwen3.8-27b-worker",
        )
        for model in models:
            for effort in ("high", "max"):
                with self.subTest(model=model, effort=effort):
                    request = {
                        "model": model,
                        "reasoning_effort": effort,
                        "messages": [{"role": "user", "content": "test"}],
                    }

                    updated = normalize_qwen38_reasoning_effort(request)

                    self.assertIsNot(updated, request)
                    self.assertEqual(updated["reasoning_effort"], "xhigh")
                    self.assertEqual(request["reasoning_effort"], effort)

    def test_qwen_reasoning_effort_preserves_supported_and_other_models(self) -> None:
        requests = [
            {
                "model": "qwen3.8-flash-next-thinking",
                "reasoning_effort": effort,
            }
            for effort in ("low", "medium", "xhigh")
        ]
        requests.append(
            {"model": "glm-5.3-flash-high", "reasoning_effort": "high"}
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertIs(normalize_qwen38_reasoning_effort(request), request)

    def test_litellm_binds_the_shared_qwen_alias_to_the_active_profile(self) -> None:
        profile_name, model = proxy._active_anthropic_model(
            {"profile": "qwen38-flash-uncensored"}
        )

        self.assertEqual(profile_name, "qwen38-flash-uncensored")
        self.assertEqual(
            model,
            "anthropic/qwen3.8-flash-next-uncensored-mxfp4-fp8",
        )

    def test_litellm_qwen_alias_uses_nonthinking_sampling_recipe(self) -> None:
        config = (ROOT / "config" / "litellm.yaml").read_text()
        block = config.split("- model_name: qwen3.8-flash-next-fast", 1)[1].split(
            "\n  - model_name:", 1
        )[0]
        for expected in (
            "model: os.environ/HOSTED_INFERENCE_OPENAI_MODEL",
            "temperature: 0.7",
            "top_p: 0.8",
            "presence_penalty: 1.5",
            "top_k: 20",
            "min_p: 0.0",
            "repetition_penalty: 1.0",
            "enable_thinking: false",
            "preserve_thinking: false",
        ):
            self.assertIn(expected, block)

    def test_litellm_qwen_thinking_alias_uses_reasoning_sampling_recipe(self) -> None:
        config = (ROOT / "config" / "litellm.yaml").read_text()
        block = config.split(
            "- model_name: qwen3.8-flash-next-thinking", 1
        )[1].split("\n  - model_name:", 1)[0]
        for expected in (
            "model: os.environ/HOSTED_INFERENCE_OPENAI_MODEL",
            "temperature: 1.0",
            "top_p: 0.95",
            "presence_penalty: 0.0",
            "top_k: 20",
            "min_p: 0.0",
            "repetition_penalty: 1.0",
            "enable_thinking: true",
            "preserve_thinking: true",
            "supports_reasoning: true",
            "supports_max_reasoning_effort: true",
        ):
            self.assertIn(expected, block)

    def test_litellm_qwen_worker_aliases_use_secondary_backend(self) -> None:
        config = (ROOT / "config" / "litellm.yaml").read_text()
        for alias, thinking in (
            ("qwen3.8-27b-workers-thinking", True),
            ("qwen3.8-27b-workers-fast", False),
        ):
            with self.subTest(alias=alias):
                block = config.split(f"- model_name: {alias}", 1)[1].split(
                    "\n  - model_name:", 1
                )[0]
                self.assertIn(
                    "model: os.environ/HOSTED_WORKER_OPENAI_MODEL", block
                )
                self.assertIn(
                    "api_base: os.environ/HOSTED_WORKER_API_BASE", block
                )
                self.assertIn(f"enable_thinking: {str(thinking).lower()}", block)
                self.assertIn("max_input_tokens: 131072", block)

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
            "deepseek-v4-flash": ("--pipeline-parallel-size", "6"),
            GLM_PROFILE: ("--quantization", "quark"),
            "qwen38-4x27b": ("--data-parallel-size", "4"),
            "qwen38-flash": ("--tensor-parallel-size", "4"),
            "qwen38-flash-uncensored": ("--tensor-parallel-size", "4"),
        }
        for name, (option, value) in expectations.items():
            with self.subTest(profile=name):
                profile = load_profile(name)
                command = build_command(
                    profile["model"],
                    profile["runtime"],
                    Path("/models") / name,
                    "127.0.0.1",
                    8000,
                )
                option_index = command.index(option)
                self.assertEqual(command[option_index + 1], value)
                if profile["runtime"].get("backend", "vllm") == "vllm":
                    self.assertEqual(
                        command[1:3], ["-m", "r9700.vllm_entrypoint"]
                    )

    def test_qwen_27b_worker_pool_uses_four_disjoint_tp1_replicas(self) -> None:
        profile = load_profile("qwen38-4x27b")
        model = profile["model"]
        runtime = profile["runtime"]
        self.assertEqual(runtime["role"], "worker-pool")
        self.assertEqual(
            runtime["parallel"],
            {
                "tensor": 1,
                "pipeline": 1,
                "data": 4,
                "enable_expert_parallel": False,
                "disable_custom_all_reduce": True,
            },
        )
        self.assertEqual(
            runtime["gpu_bdfs"],
            [
                "0000:07:00.0",
                "0000:0a:00.0",
                "0000:23:00.0",
                "0000:e6:00.0",
            ],
        )
        primary_bdfs = {
            bdf.lower()
            for bdf in load_profile("qwen38-flash-uncensored")["runtime"][
                "gpu_bdfs"
            ]
        }
        self.assertFalse(primary_bdfs.intersection(runtime["gpu_bdfs"]))
        self.assertEqual(model["vllm"]["quantization"], "quark")
        self.assertTrue(model["supports_dflash"])
        self.assertEqual(len(model["auxiliary_artifacts"]), 2)
        drafter = model["auxiliary_artifacts"][0]
        self.assertEqual(
            drafter["repository"], "syvai/Qwen3.8-27B-DFlash2-W4A16"
        )
        self.assertEqual(
            drafter["revision"], "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"
        )
        speculative = runtime["speculative_config"]
        self.assertEqual(speculative["method"], "dflash")
        self.assertEqual(
            speculative["model_artifact"], "qwen38-27b-dflash2-w4a16"
        )
        self.assertEqual(speculative["num_speculative_tokens"], 4)
        self.assertEqual(speculative["draft_tensor_parallel_size"], 1)
        self.assertEqual(speculative["attention_backend"], "TRITON_ATTN")
        self.assertEqual(speculative["draft_sample_method"], "probabilistic")
        self.assertEqual(runtime["cache"]["dtype"], "fp8")
        self.assertEqual(
            runtime["cache"]["prefix_cache_retention_interval"], 1616
        )
        self.assertEqual(speculative["kv_cache_dtype"], "fp8")
        self.assertIn("0018", runtime["required_patches"])
        self.assertTrue(runtime["cache"]["prefix_cache"])
        self.assertEqual(runtime["environment"]["VLLM_KV_CACHE_LAYOUT"], "LBHNC")

        command = build_command(
            model, runtime, Path("/models/qwen38-27b"), "127.0.0.1", 8100
        )
        for option, value in (
            ("--tensor-parallel-size", "1"),
            ("--data-parallel-size", "4"),
            ("--quantization", "quark"),
            ("--max-model-len", "131072"),
        ):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--language-model-only", command)
        speculative_value = json.loads(
            command[command.index("--speculative-config") + 1]
        )
        self.assertNotIn("model_artifact", speculative_value)
        self.assertEqual(
            speculative_value["model"],
            "/mnt/ai/models/qwen/Qwen3.8-27B-DFlash2-W4A16",
        )

    def test_qwen_multi_pins_primary_and_worker_components(self) -> None:
        profile = load_profile("qwen-multi")
        primary = load_profile("qwen38-flash-uncensored")
        workers = load_profile("qwen38-4x27b")
        components = profile["components"]

        self.assertEqual(profile["model"]["name"], primary["model"]["name"])
        self.assertEqual(profile["runtime"]["name"], primary["runtime"]["name"])
        self.assertEqual(
            components["primary"]["profile_sha256"], primary["_sha256"]
        )
        self.assertEqual(
            components["worker_pool"]["profile_sha256"], workers["_sha256"]
        )
        self.assertEqual(
            set(profile["stack"]["litellm_aliases"]),
            set(primary["stack"]["litellm_aliases"])
            | set(workers["stack"]["litellm_aliases"]),
        )
        self.assertTrue(
            launcher._state_matches_target(
                {
                    "profile": "qwen38-flash-uncensored",
                    "model": primary["model"]["name"],
                    "runtime": primary["runtime"]["name"],
                    "runtime_mode": None,
                },
                profile_name="qwen-multi",
                model_name=profile["model"]["name"],
                runtime_name=profile["runtime"]["name"],
                runtime_mode=None,
                compatible_profiles={"qwen38-flash-uncensored"},
            )
        )

    def test_qwen_multi_defines_one_active_lead_and_four_workers(self) -> None:
        stack = load_profile("qwen-multi")["stack"]
        settings = stack["claude_settings"]
        agents = stack["claude_agents"]

        self.assertEqual(settings["teammateMode"], "in-process")
        self.assertEqual(
            settings["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"], "1"
        )
        self.assertEqual(
            settings["env"]["ANTHROPIC_MODEL"],
            "qwen3.8-flash-next-thinking",
        )
        self.assertEqual(
            set(agents),
            {
                "qwen-team-lead",
                "qwen-worker-explorer",
                "qwen-worker-implementer-a",
                "qwen-worker-implementer-b",
                "qwen-worker-verifier",
            },
        )
        self.assertEqual(
            agents["qwen-team-lead"]["model"],
            "qwen3.8-flash-next-thinking",
        )
        for name in (
            "qwen-worker-implementer-a",
            "qwen-worker-implementer-b",
            "qwen-worker-verifier",
        ):
            self.assertEqual(
                agents[name]["model"], "qwen3.8-27b-workers-thinking"
            )
        self.assertEqual(
            agents["qwen-worker-explorer"]["model"],
            "qwen3.8-27b-workers-fast",
        )
        self.assertNotIn("Edit", agents["qwen-worker-explorer"]["tools"])
        self.assertNotIn("Edit", agents["qwen-worker-verifier"]["tools"])

    def test_qwen_multi_stop_plan_includes_all_three_services(self) -> None:
        with patch("r9700.launcher._run") as run:
            launcher.stop(profile_name="qwen-multi", dry_run=True)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("stop-litellm-proxy", commands[0][0])
        self.assertEqual(commands[1][1:3], ["worker-pool", "stop"])
        self.assertIn("stop-r9700-runtime", commands[2][0])

    def test_primary_launcher_refuses_worker_pool_profile(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "secondary worker pool"):
            launcher._target("qwen38-4x27b")

    def test_worker_pool_start_command_is_persistent_and_graceful(self) -> None:
        command = worker_pool._start_command(
            "qwen38-4x27b", host="127.0.0.1", port=8100, ready_timeout=1200
        )
        self.assertEqual(command[:4], ["systemd-run", "--user", "--unit", worker_pool.UNIT])
        self.assertIn("--property=KillMode=control-group", command)
        self.assertIn("--property=KillSignal=SIGINT", command)
        self.assertIn("--property=SendSIGKILL=no", command)
        self.assertNotIn("SIGKILL", " ".join(command).replace("SendSIGKILL=no", ""))

    def test_worker_pool_rejects_overlap_with_managed_primary(self) -> None:
        profile = load_profile("qwen38-4x27b")
        overlapping = json.loads(json.dumps(profile))
        overlapping["runtime"]["gpu_bdfs"][0] = load_profile(
            "qwen38-flash-uncensored"
        )["runtime"]["gpu_bdfs"][0]
        with patch(
            "r9700.worker_pool.runtime_service.managed_state",
            return_value={"profile": "qwen38-flash-uncensored"},
        ):
            with self.assertRaisesRegex(ConfigurationError, "overlaps"):
                worker_pool._assert_disjoint_from_primary(overlapping)

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

    def test_glm_dflash_is_identity_bound_and_enabled_by_default(self) -> None:
        profile = load_profile(GLM_PROFILE)
        artifact = profile["model"]["auxiliary_artifacts"][0]
        self.assertEqual(artifact["repository"], "incoai/GLM-5.3-Flash-DFlash2")
        self.assertEqual(artifact["revision"][:7], "bf582e4")

        default_runtime = profile["runtime"]
        speculative = default_runtime["speculative_config"]
        self.assertEqual(default_runtime["limits"]["max_model_len"], 524288)
        self.assertEqual(default_runtime["cache"]["dtype"], "fp8")
        self.assertEqual(speculative["model_artifact"], "dflash2-drafter")
        self.assertEqual(speculative["method"], "dflash")
        # K4 is the stable deployment setting for this rollback.
        self.assertEqual(speculative["num_speculative_tokens"], 4)
        self.assertEqual(speculative["draft_tensor_parallel_size"], 8)
        self.assertEqual(speculative["attention_backend"], "TRITON_ATTN")
        self.assertEqual(speculative["kv_cache_dtype"], "fp8")
        self.assertEqual(
            default_runtime["environment"]["VLLM_ROCM_USE_TRITON_MXFP4_GEMV"],
            "1",
        )

        self.assertEqual(
            default_runtime["required_patches"][-17:],
            [
                "0009",
                "0010",
                "0011",
                "0012",
                "0013",
                "0014",
                "0015",
                "0016",
                "0017",
                "0018",
                "0019",
                "0020",
                "0021",
                "0022",
                "0023",
                "0024",
                "0025",
            ],
        )
        self.assertTrue(
            {
                "0003",
                "0005",
                "0006",
                "0007",
                "0008",
                "0010",
                "0011",
                "0012",
                "0013",
                "0014",
                "0015",
                "0016",
                "0017",
                "0018",
                "0019",
                "0020",
                "0021",
                "0022",
                "0023",
                "0024",
                "0025",
            }.issubset(default_runtime["required_patches"])
        )

    def test_glm_gluon_sparse_mla_stays_isolated_and_opt_in(self) -> None:
        profile = load_profile(
            str(ROOT / "profiles/dev/glm53-flash-rocm10-gluon.json")
        )
        runtime = profile["runtime"]
        self.assertEqual(profile["status"], "development")
        self.assertEqual(runtime["status"], "diagnostic-only")
        self.assertEqual(runtime["recipe"], "vllm_glm53flashrocm10_v0.30")
        self.assertEqual(runtime["environment"]["VLLM_ROCM_USE_GLUON_SPARSE_MLA"], "1")
        self.assertTrue({"0026", "1001"}.issubset(runtime["required_patches"]))

        production = load_profile(GLM_PROFILE)["runtime"]
        self.assertEqual(production["recipe"], "vllm_glm53flashrocm10_v0.29")
        self.assertNotIn("VLLM_ROCM_USE_GLUON_SPARSE_MLA", production["environment"])

    def test_glm_v029_mxfp4_gemv_512k_vision_is_the_default(self) -> None:
        profile = load_profile(GLM_PROFILE)
        runtime = profile["runtime"]
        self.assertEqual(profile["status"], "production-ready")
        self.assertEqual(runtime["status"], "production-ready")
        self.assertEqual(runtime["recipe"], "vllm_glm53flashrocm10_v0.29")
        self.assertNotIn("VLLM_ROCM_MXFP4_GEMV_BLOCK_N", runtime["environment"])
        self.assertNotIn("0026", runtime["required_patches"])
        self.assertNotIn("experimental_modes", runtime)
        self.assertEqual(runtime["limits"]["max_model_len"], 524288)
        self.assertEqual(runtime["limits"]["max_num_batched_tokens"], 4096)
        self.assertEqual(runtime["limits"]["kv_cache_memory_bytes"], 4960000000)
        self.assertFalse(runtime["multimodal"]["language_model_only"])
        self.assertEqual(
            runtime["multimodal"]["encoder_attention_backend"], "TRITON_ATTN"
        )
        self.assertEqual(runtime["multimodal"]["encoder_tp_mode"], "weights")
        self.assertEqual(
            runtime["multimodal"]["limit_per_prompt"], {"image": 8, "video": 0}
        )
        command = build_command(
            profile["model"], runtime, Path("/models/glm"), "127.0.0.1", 8000
        )
        self.assertEqual(
            command[command.index("--max-model-len") + 1], "524288"
        )
        self.assertEqual(
            command[command.index("--max-num-batched-tokens") + 1], "4096"
        )
        self.assertIn("--enable-prefix-caching", command)
        self.assertEqual(
            command[command.index("--prefix-cache-retention-interval") + 1],
            "1280",
        )
        self.assertIn("--enable-prompt-tokens-details", command)
        self.assertNotIn("--kv-transfer-config", command)
        self.assertNotIn("--language-model-only", command)
        self.assertEqual(
            command[command.index("--mm-encoder-attn-backend") + 1],
            "TRITON_ATTN",
        )
        self.assertEqual(
            profile["stack"]["claude_settings"]["env"][
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
            ],
            "524288",
        )

    def test_uncensored_glm_uses_the_stable_v029_runtime(self) -> None:
        profile = load_profile("glm53-flash-uncensored")
        model = profile["model"]
        runtime = profile["runtime"]

        self.assertEqual(
            model["default_directory"],
            "/mnt/ai/models/glm/GLM-5.3-Flash-UNCENSORED-Quark-MXFP4",
        )
        self.assertEqual(model["vllm"]["quantization"], "quark")
        self.assertEqual(runtime["recipe"], "vllm_glm53flashrocm10_v0.29")
        self.assertNotIn("experimental_modes", runtime)
        self.assertEqual(runtime["limits"]["max_model_len"], 524288)
        self.assertEqual(runtime["limits"]["max_num_batched_tokens"], 4096)
        self.assertTrue(runtime["cache"]["prefix_cache"])
        self.assertEqual(
            runtime["cache"]["prefix_cache_retention_interval"], 1280
        )
        self.assertEqual(runtime["cache"]["dtype"], "fp8")
        self.assertEqual(runtime["cache"]["cpu_offload_gb"], 0)
        self.assertNotIn("kv_transfer_config", runtime)
        self.assertTrue(runtime["enable_prompt_tokens_details"])
        self.assertEqual(runtime["speculative_config"]["method"], "dflash")
        self.assertEqual(runtime["speculative_config"]["num_speculative_tokens"], 4)
        self.assertFalse(runtime["multimodal"]["language_model_only"])

        command = build_command(
            model, runtime, Path("/models/glm-uncensored"), "127.0.0.1", 8000
        )
        self.assertIn("--quantization", command)
        self.assertEqual(command[command.index("--quantization") + 1], "quark")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "glm-5.3-flash-uncensored-quark-mxfp4",
        )
        self.assertIn("--enable-prefix-caching", command)
        self.assertEqual(
            command[command.index("--prefix-cache-retention-interval") + 1],
            "1280",
        )
        self.assertNotIn("--no-enable-prefix-caching", command)
        self.assertIn("--enable-prompt-tokens-details", command)
        self.assertNotIn("--kv-transfer-config", command)
        self.assertEqual(
            profile["stack"]["claude_settings"]["env"][
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
            ],
            "524288",
        )

    def test_glm_production_profile_has_a_matching_human_summary(self) -> None:
        for profile_name in (GLM_PROFILE, "glm53-flash-uncensored"):
            profile = load_profile(profile_name)
            runtime = profile["runtime"]
            summary = (
                ROOT / f"profiles/production/{profile_name}.md"
            ).read_text()
            expected_values = (
                profile["model"]["name"],
                profile["model"]["default_directory"],
                runtime["name"],
                runtime["recipe"],
                str(runtime["limits"]["max_model_len"]),
                str(runtime["limits"]["max_num_batched_tokens"]),
                str(runtime["limits"]["kv_cache_memory_bytes"]),
            )
            for value in expected_values:
                with self.subTest(profile=profile_name, value=value):
                    self.assertIn(value, summary)

    def test_glm_long_context_safety_patches_cover_page_sizes_and_bounds(
        self,
    ) -> None:
        kernel_pages = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0011-fix-advertise-actual-GLM-kpool-kernel-page-sizes.patch"
        ).read_text()
        self.assertIn(
            "return [4 * page_size for page_size in PAGED_MQA_PAGE_SIZES]",
            kernel_pages,
        )
        self.assertIn("def get_attn_backend(self):", kernel_pages)

        slot_guards = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0012-fix-guard-block-table-loads-in-slot-mapping.patch"
        ).read_text()
        self.assertEqual(
            slot_guards.count("in_range = block_indices < block_table_stride"),
            2,
        )
        self.assertIn("mask=mask & is_local & in_range", slot_guards)
        self.assertIn("mask=is_local & in_range", slot_guards)

        compressed_workspace = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0014-fix-compress-the-GLM-indexer-decode-workspace.patch"
        ).read_text()
        self.assertIn(
            "cdiv(vllm_config.model_config.max_model_len, self.index_kpool)",
            compressed_workspace,
        )

        sharded_dflash_projection = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0015-fix-shard-the-DFlash-auxiliary-projection.patch"
        ).read_text()
        self.assertIn("self.fc = RowParallelLinear(", sharded_dflash_projection)
        self.assertIn("input_is_parallel=False", sharded_dflash_projection)
        self.assertIn("reduce_results=True", sharded_dflash_projection)

        staged_ocp_mx = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0016-fix-stage-OCP-MX-expert-dequantization.patch"
        ).read_text()
        self.assertIn("def _prepare_w1_for_gemm(", staged_ocp_mx)
        self.assertIn("def _prepare_w2_for_gemm(", staged_ocp_mx)
        self.assertIn("del w1_gemm", staged_ocp_mx)
        self.assertIn("w2_gemm = self._prepare_w2_for_gemm", staged_ocp_mx)

        fp8_sparse_mla = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0018-fix-route-RDNA4-FP8-sparse-MLA-through-Triton.patch"
        ).read_text()
        self.assertIn(
            "standard_fp8_cache and fp8_triton_supported", fp8_sparse_mla
        )
        self.assertIn(
            "kv.to(tl.float32) * tl.load(kv_scale_ptr)", fp8_sparse_mla
        )
        self.assertIn(
            "RDNA4 Triton FP8 sparse MLA requires a BF16 query",
            fp8_sparse_mla,
        )

        persistent_prefill = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0019-fix-disable-persistent-sparse-MLA-for-prefill-continuations.patch"
        ).read_text()
        self.assertIn("is_chunked_continuation", persistent_prefill)
        self.assertIn("if not use_triton_sparse and use_persistent", persistent_prefill)

        full_bf16_sparse_mla = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0020-fix-route-RDNA4-full-BF16-sparse-MLA-through-Triton.patch"
        ).read_text()
        self.assertIn("rdna4_triton_supported", full_bf16_sparse_mla)
        self.assertIn("SAME_QK_V", full_bf16_sparse_mla)
        self.assertIn("BLOCK_V", full_bf16_sparse_mla)

        rdna4_shuffled_kpool_decode = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0021-fix-read-shuffled-RDNA4-kpool-cache.patch"
        ).read_text()
        self.assertIn(
            "def _rdna4_fp8_paged_mqa_logits_kernel(",
            rdna4_shuffled_kpool_decode,
        )
        self.assertIn(
            "shuffled_k_offsets = (", rdna4_shuffled_kpool_decode
        )
        self.assertIn(
            "if _ON_RDNA4 and block_size > 1:",
            rdna4_shuffled_kpool_decode,
        )

        router_dedup = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0023-perf-avoid-duplicate-GLM-MoE-router-GEMM.patch"
        ).read_text()
        self.assertIn("router_logits=hidden_states", router_dedup)
        self.assertIn(
            "-        router_logits, _ = self.gate(hidden_states)", router_dedup
        )
        self.assertNotIn(
            "+        router_logits, _ = self.gate(hidden_states)", router_dedup
        )

        required_first_tools = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0024-fix-render-GLM-tool-schemas-required-first.patch"
        ).read_text()
        self.assertIn(
            "def reorder_properties_required_first(schema: Any) -> Any:",
            required_first_tools,
        )
        self.assertIn("reorder_tool_schema_required_first = True", required_first_tools)

        strict_tool_order = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0025-fix-align-GLM-strict-tool-schema-order.patch"
        ).read_text()
        self.assertIn("def reorder_tools_required_first", strict_tool_order)
        self.assertIn("tools = reorder_tools_required_first(tools)", strict_tool_order)

        bounded_renderer_warmup = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0027-fix-bound-renderer-warmup-to-prefill-budget.patch"
        ).read_text()
        self.assertIn(
            "seq_len = min(seq_len, scheduler_config.max_num_batched_tokens)",
            bounded_renderer_warmup,
        )
        self.assertIn(
            "scheduler_config=self.config.scheduler_config",
            bounded_renderer_warmup,
        )

        mamba_retirement = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0028-fix-retire-Mamba-states-across-null-gaps.patch"
        ).read_text()
        self.assertIn("self._num_retired_blocks", mamba_retirement)
        self.assertIn("if blocks[i].is_null:", mamba_retirement)
        self.assertIn("continue", mamba_retirement)

        mamba_resume_index = (
            ROOT
            / "patches/vllm_glm53flashrocm10_v0.29/"
            "0029-fix-seed-Mamba-state-index-in-Mamba-blocks.patch"
        ).read_text()
        self.assertIn(
            "// self.cache_config.mamba_block_size", mamba_resume_index
        )

    def test_glm_v029_dev_profile_keeps_explicit_diagnostic_modes(self) -> None:
        baseline = load_runtime(GLM_V029_EXPERIMENTS)
        self.assertEqual(
            set(baseline["experimental_modes"]),
            {
                "mxfp4-gemv-dflash2-k7-256k",
                "mxfp4-gemv-dflash2-k7-fp8-1m-tp8-noep",
                "mxfp4-gemv-dflash2-k3-fp8-1m-tp8-noep",
                "mxfp4-gemv-dflash2-k4-fp8-1m-bt2048-async",
                "mxfp4-gemv-dflash2-k4-fp8-1m-bt1024",
                "mxfp4-gemv-dflash2-k4-fp8-768k-vision-weights",
                "mxfp4-gemv-dflash2-k2-fp8-256k-c4-tp8-noep",
                "mxfp4-gemv-dflash2-k2-fp8-256k-c4-tp8-noep-async",
            },
        )

        async_scheduler = load_runtime(
            GLM_V029_EXPERIMENTS,
            "mxfp4-gemv-dflash2-k4-fp8-1m-bt2048-async",
        )
        self.assertTrue(async_scheduler["scheduler"]["async"])
        self.assertTrue(async_scheduler["scheduler"]["enforce_eager"])
        self.assertEqual(async_scheduler["limits"], baseline["limits"])
        self.assertEqual(
            async_scheduler["speculative_config"],
            baseline["speculative_config"],
        )

        bt1024 = load_runtime(
            GLM_V029_EXPERIMENTS, "mxfp4-gemv-dflash2-k4-fp8-1m-bt1024"
        )
        self.assertEqual(bt1024["limits"]["max_model_len"], 1048576)
        self.assertEqual(bt1024["limits"]["max_num_seqs"], 1)
        self.assertEqual(bt1024["limits"]["max_num_batched_tokens"], 1024)
        self.assertEqual(bt1024["speculative_config"]["num_speculative_tokens"], 4)
        self.assertEqual(bt1024["cache"]["dtype"], "fp8")

        vision = load_runtime(
            GLM_V029_EXPERIMENTS,
            "mxfp4-gemv-dflash2-k4-fp8-768k-vision-weights",
        )
        self.assertEqual(vision["limits"]["max_model_len"], 786432)
        self.assertEqual(vision["limits"]["max_num_batched_tokens"], 4096)
        self.assertEqual(vision["limits"]["kv_cache_memory_bytes"], 4960000000)
        self.assertFalse(vision["multimodal"]["language_model_only"])
        self.assertEqual(
            vision["multimodal"]["encoder_attention_backend"],
            "TRITON_ATTN",
        )
        self.assertEqual(vision["multimodal"]["encoder_tp_mode"], "weights")
        self.assertEqual(
            vision["multimodal"]["limit_per_prompt"],
            {"image": 8, "video": 0},
        )
        self.assertEqual(
            vision["multimodal"]["processor_kwargs"]["max_image_tokens"],
            4096,
        )

        profile = load_profile(GLM_PROFILE)
        vision_command = build_command(
            profile["model"], vision, Path("/models/glm"), "127.0.0.1", 8000
        )
        self.assertNotIn("--language-model-only", vision_command)
        self.assertEqual(
            vision_command[vision_command.index("--mm-encoder-attn-backend") + 1],
            "TRITON_ATTN",
        )
        self.assertEqual(
            vision_command[vision_command.index("--mm-encoder-tp-mode") + 1],
            "weights",
        )

        fallback = load_runtime(
            GLM_V029_EXPERIMENTS, "mxfp4-gemv-dflash2-k7-256k"
        )
        self.assertEqual(fallback["limits"]["max_model_len"], 262144)
        self.assertNotIn("kv_cache_memory_bytes", fallback["limits"])
        self.assertEqual(fallback["cache"]["dtype"], "bfloat16")
        self.assertTrue(fallback["parallel"]["enable_expert_parallel"])
        self.assertEqual(
            fallback["speculative_config"]["kv_cache_dtype"], "bfloat16"
        )
        self.assertEqual(
            fallback["environment"]["VLLM_ROCM_USE_TRITON_MXFP4_GEMV"],
            "1",
        )

        tp_noep = load_runtime(
            GLM_V029_EXPERIMENTS,
            "mxfp4-gemv-dflash2-k7-fp8-1m-tp8-noep",
        )
        self.assertEqual(tp_noep["parallel"]["tensor"], 8)
        self.assertFalse(tp_noep["parallel"]["enable_expert_parallel"])
        self.assertEqual(tp_noep["limits"]["max_model_len"], 1048576)
        self.assertEqual(tp_noep["limits"]["max_num_batched_tokens"], 512)
        self.assertEqual(tp_noep["cache"]["dtype"], "fp8")
        self.assertEqual(
            tp_noep["speculative_config"]["num_speculative_tokens"], 7
        )

        k3 = load_runtime(
            GLM_V029_EXPERIMENTS, "mxfp4-gemv-dflash2-k3-fp8-1m-tp8-noep"
        )
        self.assertEqual(k3["parallel"], baseline["parallel"])
        self.assertEqual(k3["limits"]["max_num_batched_tokens"], 512)
        self.assertEqual(k3["cache"], baseline["cache"])
        self.assertEqual(k3["speculative_config"]["num_speculative_tokens"], 3)

        workflow_c4 = load_runtime(
            GLM_V029_EXPERIMENTS,
            "mxfp4-gemv-dflash2-k2-fp8-256k-c4-tp8-noep",
        )
        self.assertEqual(workflow_c4["parallel"], baseline["parallel"])
        self.assertEqual(workflow_c4["limits"]["max_model_len"], 262144)
        self.assertEqual(workflow_c4["limits"]["max_num_seqs"], 4)
        self.assertEqual(
            workflow_c4["limits"]["max_num_batched_tokens"], 512
        )
        self.assertEqual(
            workflow_c4["limits"]["kv_cache_memory_bytes"], 6591622400
        )
        self.assertEqual(workflow_c4["cache"]["dtype"], "fp8")
        self.assertEqual(
            workflow_c4["speculative_config"]["num_speculative_tokens"], 2
        )

        workflow_c4_async = load_runtime(
            GLM_V029_EXPERIMENTS,
            "mxfp4-gemv-dflash2-k2-fp8-256k-c4-tp8-noep-async",
        )
        self.assertTrue(workflow_c4_async["scheduler"]["async"])
        self.assertTrue(workflow_c4_async["scheduler"]["enforce_eager"])
        for key in ("parallel", "limits", "cache", "speculative_config"):
            self.assertEqual(workflow_c4_async[key], workflow_c4[key])

    def test_rocm10_glm_defaults_to_v029_gemv_fp8_512k_vision(self) -> None:
        profile = load_profile(GLM_PROFILE)
        baseline = profile["runtime"]
        self.assertEqual(baseline["speculative_config"]["method"], "dflash")
        self.assertEqual(
            baseline["speculative_config"]["num_speculative_tokens"], 4
        )
        self.assertEqual(
            baseline["speculative_config"]["kv_cache_dtype"], "fp8"
        )
        self.assertEqual(baseline["limits"]["max_model_len"], 524288)
        self.assertEqual(baseline["limits"]["max_num_seqs"], 1)
        self.assertEqual(
            baseline["limits"]["max_num_batched_tokens"], 4096
        )
        self.assertEqual(
            baseline["limits"]["gpu_memory_utilization"], 0.995
        )
        self.assertEqual(
            baseline["limits"]["kv_cache_memory_bytes"], 4960000000
        )
        self.assertEqual(baseline["cache"]["dtype"], "fp8")
        self.assertFalse(baseline["multimodal"]["language_model_only"])
        self.assertNotIn("VLLM_ROCM_MXFP4_GEMV_BLOCK_N", baseline["environment"])

        command = build_command(
            profile["model"],
            baseline,
            Path("/models/glm"),
            "127.0.0.1",
            8000,
        )
        self.assertEqual(
            command[command.index("--max-model-len") + 1], "524288"
        )
        self.assertEqual(
            command[command.index("--kv-cache-dtype") + 1], "fp8"
        )
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1],
            "4960000000",
        )

    def test_vllm_speculative_model_resolves_from_identity_bound_artifact(self) -> None:
        profile = load_profile(GLM_PROFILE)
        runtime = profile["runtime"]
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

    def test_glm_is_mrv2_with_native_prefix_cache(self) -> None:
        profile = load_profile(GLM_PROFILE)
        runtime = profile["runtime"]
        self.assertTrue(runtime["cache"]["prefix_cache"])
        self.assertEqual(
            runtime["cache"]["prefix_cache_retention_interval"], 1280
        )
        self.assertTrue(runtime["enable_prompt_tokens_details"])
        self.assertEqual(runtime["cache"]["cpu_offload_gb"], 0)
        self.assertEqual(runtime["parallel"]["tensor"], 8)
        self.assertEqual(runtime["limits"]["max_model_len"], 524288)
        self.assertEqual(runtime["cache"]["dtype"], "fp8")
        self.assertEqual(runtime["speculative_config"]["method"], "dflash")
        self.assertEqual(
            runtime["speculative_config"]["num_speculative_tokens"], 4
        )
        self.assertEqual(runtime["environment"]["VLLM_USE_V2_MODEL_RUNNER"], "1")
        self.assertEqual(runtime["moe_backend"], "emulation")
        self.assertEqual(runtime["linear_backend"], "emulation")

    def test_claude_settings_disable_dynamic_reminders(self) -> None:
        profile_paths = [
            *(ROOT / "profiles" / "production").glob("*.json"),
            *(ROOT / "profiles" / "dev").glob("*.json"),
        ]
        for path in profile_paths:
            with self.subTest(profile=path.stem):
                settings = load_profile(str(path))["stack"]["claude_settings"]
                self.assertEqual(settings["totalTokensReminder"], "off")
                self.assertEqual(
                    settings["env"]["CLAUDE_CODE_TODO_REMINDER_MODE"], "off"
                )
                self.assertNotIn(
                    "CLAUDE_CODE_TOTAL_TOKENS_REMINDER", settings["env"]
                )

        template_paths = (ROOT / "templates" / ".claude").glob(
            "*/*.settings.local.json"
        )
        for path in template_paths:
            with self.subTest(template=str(path.relative_to(ROOT))):
                settings = json.loads(path.read_text())
                self.assertEqual(settings["totalTokensReminder"], "off")
                self.assertEqual(
                    settings["env"]["CLAUDE_CODE_TODO_REMINDER_MODE"], "off"
                )
                self.assertNotIn(
                    "CLAUDE_CODE_TOTAL_TOKENS_REMINDER", settings["env"]
                )

    def test_glm_model_download_includes_its_chat_template(self) -> None:
        model = load_profile(GLM_PROFILE)["model"]
        self.assertIn("chat_template.jinja", model["required_files"])
        self.assertIn("chat_template.jinja", model["allow_patterns"])

    def test_glm_api_gate_keeps_reasoning_parser_enabled(self) -> None:
        model = load_profile(GLM_PROFILE)["model"]
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

    def test_launcher_cli_defaults_to_the_full_stack(self) -> None:
        command_parser = cli.parser()
        start_args = command_parser.parse_args(
            ["launcher", "start", "glm53-flash"]
        )
        stop_args = command_parser.parse_args(["launcher", "stop"])
        runtime_only_args = command_parser.parse_args(
            ["launcher", "start", "glm53-flash", "--runtime-only"]
        )

        self.assertTrue(start_args.with_litellm)
        self.assertTrue(stop_args.with_litellm)
        self.assertFalse(runtime_only_args.with_litellm)

    def test_launcher_start_never_replaces_a_running_profile(self) -> None:
        state = {
            "profile": GLM_PROFILE,
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
                ConfigurationError, "omit --runtime-only"
            ):
                launcher.stop(with_litellm=False)

        run.assert_not_called()

    def test_launcher_dry_run_uses_transactional_stack_manager(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start("deepseek-v4-flash", dry_run=True)

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), launcher.STACK_SCRIPT)
        self.assertEqual(
            command[1:4], ["start", "--preset", "deepseek-v4-flash"]
        )
        self.assertIn("--dry-run", command)

    def test_launcher_runtime_only_uses_component_lifecycle_script(self) -> None:
        with (
            patch("r9700.launcher.managed_state", return_value=None),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            launcher.start(
                "deepseek-v4-flash", with_litellm=False, dry_run=True
            )

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), launcher.START_SCRIPT)
        self.assertEqual(command[1:3], ["--profile", "deepseek-v4-flash"])
        self.assertIn("--dry-run", command)

    def test_launcher_runtime_only_switch_stops_before_starting(self) -> None:
        state = {
            "profile": GLM_PROFILE,
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
            launcher.switch("qwen38-flash", with_litellm=False)

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
            launcher.start("qwen38-flash", dry_run=True)

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), launcher.STACK_SCRIPT)
        self.assertEqual(command[1:4], ["start", "--preset", "qwen38-flash"])
        self.assertIn("--proxy-ready-timeout", command)
        self.assertIn("--dry-run", command)

    def test_launcher_rejects_modes_in_a_production_profile(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "unknown experimental runtime mode",
        ):
            launcher.start(
                GLM_PROFILE,
                runtime_mode="mxfp4-gemv-dflash2-k7-256k",
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
            launcher.start("qwen38-flash")

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
            "profile": GLM_PROFILE,
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
            launcher.switch("qwen38-flash")

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(Path(commands[0][0]), launcher.STACK_SCRIPT)
        self.assertEqual(commands[0][1], "stop")
        self.assertEqual(Path(commands[1][0]), launcher.STACK_SCRIPT)
        self.assertEqual(commands[1][1:4], ["start", "--preset", "qwen38-flash"])

    def test_launcher_validates_stack_switch_before_stopping(self) -> None:
        state = {"profile": GLM_PROFILE}
        with (
            patch("r9700.launcher.managed_state", return_value=state),
            patch("r9700.launcher.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(
                ConfigurationError, "only with --runtime-only"
            ):
                launcher.switch(
                    "qwen38-flash",
                    host="127.0.0.1",
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

    def test_model_verification_accepts_bound_huggingface_tree_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            shard = destination / "model.safetensors"
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
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
            revision = "2" * 40
            tree_path = destination / ".cache/huggingface/trees" / f"{revision}.json"
            tree_path.parent.mkdir(parents=True)
            tree_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "files": {
                            shard.name: {"size": shard.stat().st_size},
                        },
                    }
                )
            )
            model = {
                "name": "fixture",
                "repository": "example/model",
                "revision": revision,
                "expected_shards": 1,
                "weight_pattern": "*.safetensors",
                "required_files": [],
                "checkpoint_weight_bytes": 4,
                "source_evidence": {
                    "kind": "huggingface_tree",
                    "path": str(tree_path.relative_to(destination)),
                    "sha256": sha256_file(tree_path),
                },
                "_sha256": "1" * 64,
            }
            with (
                patch("r9700.models.load_model", return_value=model),
                patch(
                    "r9700.models.resolve_model_directory",
                    return_value=destination,
                ),
            ):
                payload = verify_model("fixture")

            self.assertEqual(payload["resolved_revision"], revision)
            self.assertEqual(payload["checkpoint"]["shard_count"], 1)

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
