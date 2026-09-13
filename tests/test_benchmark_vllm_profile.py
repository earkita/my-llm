import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark-vllm-profile.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_vllm_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)

CLIENT_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "vllm-bench-client.py"
)
CLIENT_SPEC = importlib.util.spec_from_file_location(
    "vllm_bench_client", CLIENT_MODULE_PATH
)
assert CLIENT_SPEC is not None and CLIENT_SPEC.loader is not None
CLIENT = importlib.util.module_from_spec(CLIENT_SPEC)
sys.modules[CLIENT_SPEC.name] = CLIENT
CLIENT_SPEC.loader.exec_module(CLIENT)


class BenchmarkVllmProfileTests(unittest.TestCase):
    def target(self):
        return BENCHMARK.Target(
            requested_profile="qwen-multi",
            component="worker-pool",
            profile_path=Path("worker.json"),
            state_path=Path("service.json"),
            endpoint="http://127.0.0.1:8100",
            model_name="worker-internal",
            served_name="worker-served",
            model_directory=Path("/models/worker"),
            model_revision="revision",
            runtime_name="worker-runtime",
            recipe="vllm_recipe",
            max_model_len=131072,
            max_num_seqs=1,
            data_parallel_size=4,
            cache_dtype="fp8",
            prefix_cache=True,
            profile_sha256="abc",
            runtime_profile_sha256="def",
            access_mode="direct",
            alias=None,
            gateway_state_path=None,
            request_overrides={},
        )

    def test_worker_command_pins_single_rank(self) -> None:
        scenario = BENCHMARK.SCENARIOS["decode"]
        command = BENCHMARK.build_command(
            Path("/runtime/bin/vllm"),
            self.target(),
            scenario,
            Path("/results"),
            2,
        )

        self.assertIn("--served-model-name", command)
        self.assertIn("worker-served", command)
        self.assertIn("X-data-parallel-rank=2", command)
        self.assertIn("--save-detailed", command)
        backend = command.index("--backend")
        endpoint = command.index("--endpoint")
        self.assertEqual(command[backend + 1], "openai-chat")
        self.assertEqual(command[endpoint + 1], "/v1/chat/completions")
        self.assertNotIn("--ignore-eos", command)
        self.assertNotIn("--temperature", command)
        self.assertEqual(command[0], "/runtime/bin/python")
        self.assertEqual(
            command[1],
            str(Path(__file__).resolve().parents[1] / "scripts" / "vllm-bench-client.py"),
        )

    def test_benchmark_client_splits_combined_choices_and_usage(self) -> None:
        messages = [
            'data: {"choices":[{"delta":{"content":"x"}}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":20}}',
            "data: [DONE]",
        ]

        split = CLIENT.split_combined_usage_chunks(messages)

        self.assertEqual(len(split), 3)
        choice = json.loads(split[0].removeprefix("data: "))
        usage = json.loads(split[1].removeprefix("data: "))
        self.assertNotIn("usage", choice)
        self.assertEqual(usage["choices"], [])
        self.assertEqual(usage["usage"]["completion_tokens"], 20)
        self.assertEqual(split[2], "data: [DONE]")

    def test_raw_command_forces_fixed_greedy_completion(self) -> None:
        target = BENCHMARK.replace(self.target(), access_mode="raw")

        command = BENCHMARK.build_command(
            Path("/runtime/bin/vllm"),
            target,
            BENCHMARK.SCENARIOS["decode"],
            Path("/results"),
            0,
        )

        backend = command.index("--backend")
        endpoint = command.index("--endpoint")
        self.assertEqual(command[backend + 1], "openai")
        self.assertEqual(command[endpoint + 1], "/v1/completions")
        self.assertIn("--ignore-eos", command)
        temperature = command.index("--temperature")
        self.assertEqual(command[temperature + 1], "0")

    def test_direct_chat_command_applies_alias_parameters(self) -> None:
        target = BENCHMARK.replace(
            self.target(),
            request_overrides={
                "reasoning_effort": "high",
                "temperature": 0.7,
            },
        )

        command = BENCHMARK.build_command(
            Path("/runtime/bin/vllm"),
            target,
            BENCHMARK.SCENARIOS["decode"],
            Path("/results"),
            0,
        )

        extra_body = command.index("--extra-body")
        self.assertEqual(
            json.loads(command[extra_body + 1]),
            {"reasoning_effort": "high", "temperature": 0.7},
        )

    def test_client_environment_forces_cpu_on_mixed_gpu_host(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VLLM_TARGET_DEVICE": "cuda",
                "HIP_VISIBLE_DEVICES": "0,1",
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            },
            clear=True,
        ):
            environment = BENCHMARK.benchmark_environment(
                Path("/repo"), Path("/recipe"), Path("/rocm")
            )

        self.assertEqual(environment["VLLM_TARGET_DEVICE"], "cpu")
        self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["VLLM_PLUGINS"], "")

    def test_pool_command_does_not_pin_rank(self) -> None:
        command = BENCHMARK.build_command(
            Path("/runtime/bin/vllm"),
            self.target(),
            BENCHMARK.SCENARIOS["pool"],
            Path("/results"),
            None,
        )

        self.assertNotIn("--header", command)
        index = command.index("--max-concurrency")
        self.assertEqual(command[index + 1], "4")

    def test_litellm_command_uses_chat_alias_without_secret(self) -> None:
        target = BENCHMARK.replace(
            self.target(),
            endpoint="http://127.0.0.1:4000",
            served_name="qwen-worker-fast",
            access_mode="litellm",
            alias="qwen-worker-fast",
            gateway_state_path=Path("proxy.json"),
        )

        command = BENCHMARK.build_command(
            Path("/runtime/bin/vllm"),
            target,
            BENCHMARK.SCENARIOS["decode"],
            Path("/results"),
            2,
        )

        backend = command.index("--backend")
        endpoint = command.index("--endpoint")
        self.assertEqual(command[backend + 1], "openai-chat")
        self.assertEqual(command[endpoint + 1], "/v1/chat/completions")
        self.assertNotIn("--header", command)
        self.assertNotIn("--temperature", command)
        self.assertNotIn("--ignore-eos", command)

    def test_summary_uses_actual_tokens_and_tpot(self) -> None:
        payload = {
            "completed": 2,
            "failed": 0,
            "input_lens": [100, 200],
            "output_lens": [51, 51],
            "ttfts": [0.5, 1.0],
            "mean_ttft_ms": 750,
            "p95_ttft_ms": 975,
            "mean_tpot_ms": 20,
            "mean_itl_ms": 19,
            "mean_e2el_ms": 1750,
            "output_throughput": 50,
            "total_token_throughput": 200,
        }
        scenario = BENCHMARK.Scenario("custom", 100, 51, 2, 1, 0, 1)

        summary = BENCHMARK.summarize_result(scenario, payload)

        self.assertEqual(summary["effective_prefill_tokens_per_second"], 200)
        self.assertEqual(summary["decode_tokens_per_second"], 50)
        self.assertEqual(summary["mean_input_tokens"], 150)
        self.assertEqual(summary["mean_output_tokens"], 51)
        self.assertEqual(summary["requested_output_tokens"], 51)
        self.assertEqual(summary["output_completion_percent"], 100)

    def test_resolves_worker_component_from_self_contained_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "profiles" / "production"
            production.mkdir(parents=True)
            worker = {
                "model": {
                    "name": "worker-internal",
                    "served_name": "worker-served",
                    "default_directory": "/models/worker",
                    "revision": "revision",
                },
                "runtime": {
                    "name": "worker-runtime",
                    "recipe": "vllm_recipe",
                    "limits": {"max_model_len": 1000, "max_num_seqs": 1},
                    "parallel": {"data": 4},
                    "cache": {"dtype": "fp8", "prefix_cache": True},
                },
            }
            combined = {
                "model": worker["model"],
                "runtime": worker["runtime"],
                "components": {
                    "worker_pool": {
                        "profile": "worker",
                        "url": "http://127.0.0.1:8100",
                    }
                },
            }
            (production / "worker.json").write_text(json.dumps(worker))
            (production / "combined.json").write_text(json.dumps(combined))

            target = BENCHMARK.resolve_target(
                root, "combined", "worker-pool", require_active=False
            )

            self.assertEqual(target.served_name, "worker-served")
            self.assertEqual(target.data_parallel_size, 4)
            self.assertEqual(target.endpoint, "http://127.0.0.1:8100")

    def test_litellm_alias_resolves_gateway_and_direct_worker_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "profiles" / "production"
            runtime = root / ".runtime"
            (runtime / "litellm").mkdir(parents=True)
            production.mkdir(parents=True)
            worker = {
                "model": {
                    "name": "worker-internal",
                    "served_name": "worker-served",
                    "default_directory": "/models/worker",
                    "revision": "revision",
                },
                "runtime": {
                    "name": "worker-runtime",
                    "recipe": "vllm_recipe",
                    "limits": {"max_model_len": 1000, "max_num_seqs": 1},
                    "parallel": {"data": 4},
                    "cache": {"dtype": "fp8", "prefix_cache": True},
                },
                "stack": {"litellm_aliases": ["worker-fast"]},
            }
            combined = {
                "name": "combined",
                "model": worker["model"],
                "runtime": worker["runtime"],
                "components": {
                    "worker_pool": {
                        "profile": "worker",
                        "url": "http://127.0.0.1:8100",
                    }
                },
                "stack": {"litellm_aliases": ["main-fast", "worker-fast"]},
            }
            (production / "worker.json").write_text(json.dumps(worker))
            (production / "combined.json").write_text(json.dumps(combined))
            (runtime / "service.json").write_text(
                json.dumps({"profile": "combined"})
            )
            (runtime / "litellm" / "service.json").write_text(
                json.dumps(
                    {
                        "profile": "combined",
                        "probe_url": "http://127.0.0.1:4000",
                    }
                )
            )

            gateway = BENCHMARK.resolve_alias_target(
                root,
                "worker-fast",
                access_mode="litellm",
                require_active=False,
            )
            direct = BENCHMARK.resolve_alias_target(
                root,
                "worker-fast",
                access_mode="direct",
                require_active=False,
            )
            raw = BENCHMARK.resolve_alias_target(
                root,
                "worker-fast",
                access_mode="raw",
                require_active=False,
            )

            self.assertEqual(gateway.component, "worker-pool")
            self.assertEqual(gateway.endpoint, "http://127.0.0.1:4000")
            self.assertEqual(gateway.served_name, "worker-fast")
            self.assertEqual(gateway.access_mode, "litellm")
            self.assertEqual(direct.endpoint, "http://127.0.0.1:8100")
            self.assertEqual(direct.served_name, "worker-served")
            self.assertEqual(direct.access_mode, "direct")
            self.assertEqual(raw.endpoint, "http://127.0.0.1:8100")
            self.assertEqual(raw.access_mode, "raw")

    def test_direct_alias_parameters_exclude_routing_and_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / ".runtime" / "litellm" / "service.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "probe_url": "http://127.0.0.1:4000",
                    }
                )
            )
            payload = {
                "data": [
                    {
                        "model_name": "model-fast",
                        "litellm_params": {
                            "model": "anthropic/raw-model",
                            "api_base": "http://127.0.0.1:8000",
                            "api_key": "secret",
                            "reasoning_effort": "high",
                            "temperature": 0.7,
                            "extra_body": {
                                "top_k": 20,
                                "chat_template_kwargs": {
                                    "enable_thinking": True
                                },
                            },
                        },
                    }
                ]
            }
            response = io.BytesIO(json.dumps(payload).encode())
            with (
                patch.object(BENCHMARK, "_litellm_key", return_value="key"),
                patch.object(
                    BENCHMARK.urllib.request,
                    "urlopen",
                    return_value=response,
                ),
            ):
                overrides = BENCHMARK._litellm_alias_overrides(
                    root, "model-fast"
                )

        self.assertEqual(overrides["reasoning_effort"], "high")
        self.assertEqual(overrides["temperature"], 0.7)
        self.assertEqual(overrides["top_k"], 20)
        self.assertNotIn("model", overrides)
        self.assertNotIn("api_base", overrides)
        self.assertNotIn("api_key", overrides)

    def test_rank_pinning_rejects_parallel_requests(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "per-worker capacity"):
            BENCHMARK.validate_scenario(
                self.target(),
                BENCHMARK.SCENARIOS["pool"],
                worker_rank=0,
            )


if __name__ == "__main__":
    unittest.main()
