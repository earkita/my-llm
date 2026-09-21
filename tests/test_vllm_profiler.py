from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r9700.backends import vllm
from r9700.config import ConfigurationError


class VllmProfilerEnvironmentTests(unittest.TestCase):
    def _runtime(self) -> dict:
        return {
            "recipe": "test_recipe",
            "parallel": {"tensor": 1, "pipeline": 1, "data": 1},
            "transport": {
                "p2p_disable": 0,
                "shm_disable": 1,
                "socket_ifname": "lo",
                "runtime_connect": 1,
                "hsa_legacy_ipc": 0,
            },
            "shutdown": {
                "worker_timeout_seconds": 60,
                "process_grace_seconds": 120,
            },
            "environment": {},
        }

    def _profiler_root(self, root: Path) -> Path:
        tool_dir = root / "lib" / "rocprofiler-sdk"
        tool_dir.mkdir(parents=True)
        (tool_dir / "librocprofiler-sdk-tool.so").touch()
        (root / "lib" / "librocprofiler-sdk.so").touch()
        (root / "lib" / "librocprofiler-sdk.so.1").touch()
        return root

    def test_resolves_profiler_libraries_from_recipe_rocm_root(self) -> None:
        runtime = self._runtime()
        runtime["profiling"] = {
            "kernel_trace": True,
            "output_path": "logs/profiles/glm53-test",
            "output_file": "glm53-%pid%",
            "delay_seconds": 180,
            "duration_seconds": 120,
            "repeat": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            rocm_home = self._profiler_root(Path(directory) / "rocm")
            with (
                patch.object(vllm, "base_environment", return_value={}),
                patch.object(vllm, "rocm_root", return_value=rocm_home),
                patch.object(vllm, "recipe_venv", return_value=Path(directory)),
                patch.object(vllm, "visible_devices", return_value=["0"]),
            ):
                result = vllm.environment(runtime)

        self.assertEqual(result["ROCPROF_KERNEL_TRACE"], "1")
        self.assertEqual(result["ROCPROF_SIGNAL_HANDLERS"], "0")
        self.assertEqual(
            result["ROCPROF_COLLECTION_PERIOD"],
            "180000000000:120000000000:1",
        )
        self.assertEqual(result["ROCPROF_OUTPUT_FILE_NAME"], "glm53-%pid%")
        self.assertIn(str(rocm_home), result["ROCP_TOOL_LIBRARIES"])
        self.assertTrue(
            result["ROCPROF_OUTPUT_PATH"].endswith("logs/profiles/glm53-test")
        )

    def test_rejects_profiler_output_outside_logs_profiles(self) -> None:
        runtime = self._runtime()
        runtime["profiling"] = {
            "kernel_trace": True,
            "output_path": "../outside",
            "duration_seconds": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            rocm_home = self._profiler_root(Path(directory) / "rocm")
            with (
                patch.object(vllm, "base_environment", return_value={}),
                patch.object(vllm, "rocm_root", return_value=rocm_home),
                patch.object(vllm, "recipe_venv", return_value=Path(directory)),
                patch.object(vllm, "visible_devices", return_value=["0"]),
            ):
                with self.assertRaises(ConfigurationError):
                    vllm.environment(runtime)


if __name__ == "__main__":
    unittest.main()
